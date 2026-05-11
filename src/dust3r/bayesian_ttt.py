
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveProcessNoise(nn.Module):
    def __init__(self, state_dim, base_log_var=-5.0):
        super().__init__()
        self.base_log_var = nn.Parameter(torch.ones(1, state_dim, 1) * base_log_var)
        self.alpha = nn.Parameter(torch.tensor(0.5))  # Sensitivity to innovation
        self.beta = nn.Parameter(torch.tensor(0.1))   # Damping factor

    def forward(self, innovation_norm=None):
        Q = torch.exp(self.base_log_var)
        if innovation_norm is not None:
            # Adaptive Q: If innovation is high (fast motion), increase Q.
            # We use a softplus to ensure positivity and smoothness.
            # Q_adapt = Q * (1 + alpha * softplus(innovation))
            factor = 1.0 + self.alpha * F.softplus(innovation_norm)
            Q = Q * factor.clamp(max=100.0)
        return Q

class UpdateController(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.log_sensitivity = nn.Parameter(torch.tensor(0.0))
        self.deadzone_coef = nn.Parameter(torch.tensor(0.2))
        self.huber_coef = nn.Parameter(torch.tensor(1.0))
        self.log_k_max = nn.Parameter(torch.tensor(0.0))
        self.scale_eps = nn.Parameter(torch.tensor(0.08))
        
        # Robust components for KITTI/Bonn
        self.motion_damper = nn.Parameter(torch.tensor(0.08))
        self.scale_gate_floor = nn.Parameter(torch.tensor(0.35))
        self.gate_floor = nn.Parameter(torch.tensor(0.05))

    def forward(self, innovation_norm, base_conf, relevance, prev_depth_median=None, curr_depth_median=None):
        sens = torch.exp(self.log_sensitivity)
        dz = self.deadzone_coef
        hb = self.huber_coef
        kmax = torch.sigmoid(self.log_k_max)
        
        # Robust Dimension Alignment for ScanNet variants.
        # Target innovation_norm shape: [B, 1, N_state].
        if base_conf.dim() == 3 and innovation_norm.dim() == 3:
            target_n = innovation_norm.shape[-1]
            # Convert [B, N, 1] -> [B, 1, N]
            if base_conf.shape[1] == target_n and base_conf.shape[-1] == 1:
                base_conf = base_conf.transpose(1, 2)
            # If still not aligned, fallback to scalar confidence per batch.
            if base_conf.shape[-1] != target_n:
                base_conf = base_conf.mean(dim=tuple(range(1, base_conf.dim())), keepdim=True)

        sigma = 1.0 / (base_conf + 1e-6)
        # Normalize innovation by observation uncertainty to make gating scale-invariant.
        norm_innovation = innovation_norm / (sigma + 1e-6)
        thr = dz
        
        # Robust Band-Pass Logic
        # 1. Deadzone (suppress small noise)
        masked_norm = torch.where(norm_innovation < thr, torch.zeros_like(norm_innovation), norm_innovation)
        
        # 2. Huber-style clamping (suppress outliers)
        delta = hb
        masked_norm = torch.sign(masked_norm) * torch.minimum(masked_norm.abs(), delta)
        masked = masked_norm * sigma
        
        # 3. Variance Inflation (modulate confidence)
        # If masked innovation is high, we reduce confidence (inflate R)
        conf_adj = base_conf / (1.0 + sens * masked.pow(2))
        
        if relevance is not None:
            conf_adj = conf_adj * relevance.clamp(min=1e-6)
            
        gate = kmax
        
        # 4. Scale Consistency Gate (for RPE improvement)
        if prev_depth_median is not None and curr_depth_median is not None:
            eps = self.scale_eps.abs()
            ratio = (curr_depth_median + 1e-6) / (prev_depth_median + 1e-6)
            # Fully soft scale gate in log-ratio space; avoids brittle on/off behavior.
            scale_err = torch.abs(torch.log(ratio.clamp(min=1e-6, max=1e6)))
            scale_score = torch.exp(-scale_err / (eps + 1e-6))
            scale_floor = self.scale_gate_floor.clamp(0.05, 0.60)
            gate = gate * torch.maximum(scale_score, scale_floor)

        # Light motion damping to avoid over-reacting on abrupt pose changes.
        if innovation_norm is not None:
            gate = gate / (1.0 + self.motion_damper.abs() * masked_norm.abs())
            # Innovation-adaptive lower bound prevents update starvation.
            gate_floor = self.gate_floor.clamp(0.005, 0.20)
            gate = torch.maximum(gate, gate_floor)
        # 0.005
        # Keep updates away from extreme 0/1 behavior.
        gate = gate.clamp(0.005, 0.90)
            
        return conf_adj, gate

class BayesianRecurrentUnit(nn.Module):
    """
    Implements the Information-Theoretic Recurrent Unit (IT-RU) for Test-Time Training.
    This module treats the recurrent state update as a Bayesian filtering process,
    incorporating both aleatoric uncertainty (from observation noise) and epistemic uncertainty
    (from state estimation).

    Key Innovations:
    1. Probabilistic State Representation: State is modeled as a Gaussian distribution N(mu, sigma).
    2. Kalman-based Feature Fusion: Optimal linear fusion based on uncertainty.
    3. Information-Theoretic Gating: Update magnitude is modulated by the KL-divergence (Information Gain).
    4. Adaptive Process Noise: Dynamically adjusts to motion magnitude.
    """
    def __init__(self, feature_dim, state_dim, hidden_dim=64):
        super().__init__()
        self.feature_dim = feature_dim
        self.state_dim = state_dim
        
        # Gating network to learn the "relevance" of the information gain
        # Input: [state_mu, state_sigma, obs_mu, obs_sigma, kl_div]
        self.gating_net = nn.Sequential(
            nn.Linear(1, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Learnable prior for initial state variance (if not provided)
        self.init_log_var = nn.Parameter(torch.zeros(1, state_dim, 1))
        self.conf_power = nn.Parameter(torch.tensor(2.5))
        self.var_floor_ratio = nn.Parameter(torch.tensor(0.20))
        self.outlier_ig_center = nn.Parameter(torch.tensor(1.5))
        self.outlier_ig_width = nn.Parameter(torch.tensor(1.0))
        
        # Adaptive Process Noise Module
        self.adaptive_process_noise = AdaptiveProcessNoise(state_dim)

    def predict_uncertainty(self, confidence_map, target_dim=None):
        """
        Estimate observation variance (R_t) from confidence map.
        R_t ~ 1 / Confidence
        If target_dim is provided (state size N), ensures output aligns with [B, 1, N] or broadcasts [B, 1, 1].
        """
        # confidence_map: [B, H, W] or [B, N] or [B, 1, N]
        
        # 1. Align dimensions to [B, N] for robust processing
        if confidence_map.dim() == 3:
            B, H, W = confidence_map.shape
            # If shape is [B, 1, N] (from model.py fix), squeeze it back to [B, N]
            if H == 1: 
                conf = confidence_map.squeeze(1)
            elif W == 1:
                conf = confidence_map.squeeze(2)
            else: 
                # [B, H, W] -> flatten -> [B, N]
                conf = confidence_map.reshape(B, -1)
        elif confidence_map.dim() == 2:
            conf = confidence_map
        else:
            # Fallback for scalar or weird shapes: average
            conf = confidence_map.mean(dim=(-1, -2) if confidence_map.dim() > 2 else -1)
            
        # 2. Robust Sharpening for Pose/KITTI
        # Suppress low-confidence regions (dynamic objects, sky, etc.)
        # Power 2.0 makes the transition from reliable to unreliable sharper.
        # This acts as a soft "hard attention" mechanism.
        conf = conf.pow(self.conf_power.clamp(1.2, 2.4))
        
        # 3. Variance Calculation
        # R = 1 / (conf + eps)
        # We clamp confidence to avoid exploding variance, but allow it to be high for uncertain regions.
        # min=1e-4 means max variance is 10000.0
        obs_var = 1.0 / (conf.clamp(min=1e-4) + 1e-6)
        
        # 4. Return [B, 1, N] for broadcasting against [B, D, N] state
        if obs_var.dim() == 2: # [B, N]
            obs_var = obs_var.unsqueeze(1) # [B, 1, N]
        elif obs_var.dim() == 1: # [B]
             obs_var = obs_var.view(-1, 1, 1) 
             
        # 5. Dimensionality Check against State
        if target_dim is not None:
            # target_dim is N (state size)
            # obs_var is [B, 1, M]
            M = obs_var.shape[-1]
            if M != target_dim and M > 1:
                # If observation variance has spatial structure (M) but doesn't match state (N),
                # we must reduce it to a scalar/global variance to allow broadcasting.
                # e.g. [B, 1, H*W] -> [B, 1, 1] -> broadcasts to [B, D, N]
                obs_var = obs_var.mean(dim=-1, keepdim=True)
                
        return obs_var

    def compute_kl_divergence(self, mu1, var1, mu2, var2):
        """
        Compute Symmetric KL (Jensen-Shannon approximation) for stability.
        Uses a robust formulation to prevent gradient explosion and handle
        high-confidence updates (small variance) gracefully.
        """
        # 1. Clamp variances for numerical stability
        v1 = var1.clamp(min=1e-6, max=1e3)
        v2 = var2.clamp(min=1e-6, max=1e3)
        
        # 2. Symmetrized KL Divergence (Jeffreys Divergence)
        # KL_sym = 0.5 * (KL(P||Q) + KL(Q||P))
        # This penalizes discrepancy relative to BOTH distributions' uncertainties.
        
        # Variance term: 0.5 * (v1/v2 + v2/v1 - 2)
        var_term = 0.5 * (v1 / v2 + v2 / v1 - 2)
        
        # Mean term (Mahalanobis): keep full quadratic sensitivity to residual discrepancies.
        mean_term = 0.5 * (mu1 - mu2).pow(2) * (1.0 / v1 + 1.0 / v2)
        
        kl = 0.5 * (var_term + mean_term)
        
        # 3. Sqrt-space compression
        # Compared with log1p, sqrt keeps stronger sensitivity in the mid-range.
        kl_compressed = torch.log1p(kl)
        
        return kl_compressed.mean(dim=1, keepdim=True) # Average over feature dimension

    def forward(self, state_mu, state_var, obs_mu, confidence_map, relevance_score=None):
        """
        Args:
            state_mu: Current state mean [B, D, N]
            state_var: Current state variance [B, D, N]
            obs_mu: New observation mean (feature) [B, D, N]
            confidence_map: Confidence of the observation
            relevance_score: Optional external relevance score (e.g., from attention)
        
        Returns:
            new_state_mu, new_state_var, info_gain
        """
        # Calculate innovation for adaptive process noise
        # Innovation = z - x
        innovation = obs_mu - state_mu
        inn_norm = innovation.abs().mean(dim=1, keepdim=True)
        
        # 0. Prediction Step: Add Adaptive Process Noise (Q)
        # Prevent variance collapse over long sequences
        # If innovation is high, Q increases to allow faster adaptation (e.g. KITTI motion)
        process_noise = self.adaptive_process_noise(inn_norm)
        state_var = state_var + process_noise

        # 1. Estimate Observation Noise (R)
        # We pass state_mu.shape[-1] (N) to ensure R aligns with state dimensions
        obs_var = self.predict_uncertainty(confidence_map, target_dim=state_mu.shape[-1])
        
        # 2. Kalman Gain Calculation
        # K = P / (P + R)
        # We assume diagonal covariance for efficiency
        kalman_gain_classic = state_var / (state_var + obs_var + 1e-6)
        # Stable gain form reduces sensitivity when P/R ratio changes abruptly.
        sqrt_p = state_var.clamp(min=1e-6).sqrt()
        sqrt_r = obs_var.clamp(min=1e-6).sqrt()
        kalman_gain_stable = sqrt_p / (sqrt_p + sqrt_r + 1e-6)
        kalman_gain = 0.80 * kalman_gain_classic + 0.20 * kalman_gain_stable
        
        # 3. Intermediate State Update (Standard EKF)
        # mu_new = mu + K * (z - mu)
        # innovation is already calculated above

        # 最新改动
        update_mu = state_mu + kalman_gain * innovation
        
        # P_new = (1 - K) * P
        update_var = (1 - kalman_gain) * state_var
        
        # 4. Information-Theoretic Gating
        # Calculate Information Gain (KL Divergence between Prior and Posterior)
        # How much did the distribution change?
        # info_gain shape: [B, 1, N]
        info_gain = self.compute_kl_divergence(update_mu, update_var, state_mu, state_var)
        
        # Learnable Gating based on Information Gain
        # If info_gain is high, it means this observation is "surprising" and informative.
        # However, if it's TOO high, it might be an outlier. 
        # The gating network allows the model to learn this non-linear relationship.
        
        # Permute for Linear layer: [B, 1, N] -> [B, N, 1]
        info_gain_permuted = info_gain.permute(0, 2, 1).clamp(max=2.5)
        adaptive_gate = self.gating_net(info_gain_permuted)
        # Permute back: [B, N, 1] -> [B, 1, N]
        adaptive_gate = adaptive_gate.permute(0, 2, 1)
        rel = relevance_score.clamp(min=0.0, max=1.0)
        # ITG ablation: replace learned gate with fixed constant and skip all subsequent modulations
        if relevance_score is not None:
            rel = relevance_score.clamp(min=0.0, max=1.0)
            adaptive_gate = adaptive_gate * (0.20 + 0.80 * rel)

            # Suppress updates when information gain is abnormally high (likely outliers).
        ig_center = F.softplus(self.outlier_ig_center) + 1e-6
        ig_width = F.softplus(self.outlier_ig_width) + 1e-6
        ig_excess = F.relu(info_gain - ig_center)
        outlier_factor = torch.exp(-((ig_excess / ig_width) ** 2))
        adaptive_gate = adaptive_gate * (0.55 + 0.45 * outlier_factor)

         # Keep adaptive gate in a stable operating range.
        adaptive_gate = adaptive_gate.clamp(min=0.04, max=0.95)
            
        # 5. Final Soft Update
        # We use the adaptive gate to interpolate between old and new state
        # This acts as a "second-order" filter on top of the Kalman filter
        new_state_mu = adaptive_gate * update_mu + (1 - adaptive_gate) * state_mu
        new_state_var = adaptive_gate * update_var + (1 - adaptive_gate) * state_var
        floor_ratio = self.var_floor_ratio.clamp(0.05, 0.40)
        new_state_var = torch.maximum(new_state_var, floor_ratio * state_var)
    
       
        return new_state_mu, new_state_var, adaptive_gate

