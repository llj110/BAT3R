<h2 align="center">BAT3R: Bayesian Adaptive Test-Time 3D Reconstruction</h2>

<h5 align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-XXXX.XXXXX-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/XXXX.XXXXX)
[![Project Page](https://img.shields.io/badge/Project-Website-33728E.svg)](https://example.com)

</h5>

<div align="center">
TL;DR: Bayesian test-time adaptation for robust long-sequence 3D reconstruction.
</div>
<br>

BAT3R extends test-time training for 3D reconstruction with a Bayesian recurrent state update mechanism, enabling robust long-sequence reconstruction with adaptive uncertainty estimation.

## News

- [2026.05] Code released.
- [2026.XX] Paper released on arXiv.

## Getting Started

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/BAT3R.git
cd BAT3R
conda create -n bat3r python=3.11 cmake=3.14.0
conda activate bat3r
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
conda install "llvm-openmp<16"
pip install evo open3d
```

Compile CUDA kernels for RoPE (from CroCo v2):
```bash
cd src/croco/models/curope/
python setup.py build_ext --inplace
cd ../../../../
```

### Download Checkpoint

Download the pretrained checkpoint from [Releases](https://github.com/YOUR_USERNAME/BAT3R/releases):
```bash
cd src
# Download cut3r_512_dpt_4_64.pth
cd ..
```

### Inference Demo

```bash
CUDA_VISIBLE_DEVICES=0 python demo.py \
    --model_path src/cut3r_512_dpt_4_64.pth \
    --seq_path examples/westlake.mp4 \
    --size 512 --output_dir tmp/output \
    --model_update_type bat3r \
    --frame_interval 1 --reset_interval 100 \
    --downsample_factor 100 --vis_threshold 6.0
```

### Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model_path` | Path to checkpoint | required |
| `--seq_path` | Input video or image folder | required |
| `--size` | Image resolution | 512 |
| `--model_update_type` | State update: `bat3r`, `ttt3r`, `none` | `bat3r` |
| `--frame_interval` | Process every N frames | 1 |
| `--reset_interval` | Reset state every N frames | 100 |
| `--vis_threshold` | Confidence threshold | 5.0 |

## Evaluation

See [eval/README.md](eval/README.md) for evaluation on 7Scenes, KITTI, Bonn, TUM, and NRGBD benchmarks.

## Method

BAT3R introduces three key components on top of the TTT3R framework:

1. **Bayesian Recurrent State Update**: Maintains state uncertainty via a Kalman-inspired update rule that adaptively weights new observations against accumulated evidence, with learnable process noise and observation variance.

2. **Global Scene Rectification**: EMA-based scene context that accumulates global features over time to rectify drift during long sequences.

3. **Pose-Delta Gate**: Temporal stabilization module that gates state updates based on relative camera motion, preventing instability during rapid viewpoint changes.

## Acknowledgements

This project is built upon the following excellent works:

- [TTT3R](https://github.com/Inception3D/TTT3R) - 3D Reconstruction as Test-Time Training
- [CUT3R](https://github.com/CUT3R/CUT3R)
- [DUSt3R](https://github.com/naver/dust3r)
- [MonST3R](https://github.com/Junyi42/monst3r)
- [CroCo](https://github.com/naver/croco)

## Citation

```bibtex
@article{bat3r2026,
  title={BAT3R: Bayesian Adaptive Test-Time 3D Reconstruction},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

## License

This project is released under the Apache License 2.0. See [LICENSE](LICENSE) for details.
