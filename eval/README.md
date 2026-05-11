# Evaluation

## Supported Benchmarks
- **7Scenes**: Indoor localization  
- **KITTI**: Outdoor driving  
- **Bonn**: Dynamic scenes  
- **TUM RGB-D**: Indoor SLAM  
- **NRGBD**: Novel view synthesis  

## Relative Pose Evaluation
```bash
cd eval/relpose
python launch.py --dataset 7scenes --model_path ../../src/cut3r_512_dpt_4_64.pth --model_update_type bat3r
```

## Multi-View Reconstruction
```bash
cd eval/mv_recon
python launch.py --dataset 7scenes --model_path ../../src/cut3r_512_dpt_4_64.pth
```

## Model Update Types
- `bat3r`: Full Bayesian adaptive TTT (recommended)
- `ttt3r`: Standard TTT update  
- `none`: No test-time adaptation (baseline)

Results are saved to `eval_results/` with per-scene metrics and PLY exports.
