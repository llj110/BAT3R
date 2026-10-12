# BAT3R: Bayesian Adaptive Test-Time 3D Reconstruction

Bayesian recurrent state adaptation for long-sequence 3D reconstruction,
built on TTT3R, CUT3R, DUSt3R, and CroCo.

## BAT3R implementation

```text
BAT3R/
├── bat3r_demo.py                 # inference and PLY export
├── bat3r_reconstruct.py          # reconstruction and PNG rendering
├── bat3r_eval.py                 # common benchmark launcher
├── bat3r_run.sh                  # portable demo launcher
├── src/bat3r/
│   ├── bat3r_model.py            # recurrent reconstruction + global/pose gates
│   ├── bat3r_update.py           # Bayesian fusion + adaptive process noise
│   └── bat3r_inference.py        # sequence inference
├── src/dust3r/                   # inherited utilities + compatibility imports
├── src/croco/                    # inherited encoder/decoder components
├── eval/                        # benchmark implementations
└── tests/                       # CPU regression checks
```

See [the code map](docs/BAT3R_CODEMAP.md) for component locations and
[the release checklist](docs/BAT3R_RELEASE_CHECKLIST.md) for remaining artifacts.

## Installation

Run commands from the repository root, using Python 3.11 and a CUDA-compatible
PyTorch installation:

```bash
git clone https://github.com/llj110/BAT3R.git
cd BAT3R
conda create -n bat3r python=3.11 cmake=3.14.0
conda activate bat3r
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
conda install "llvm-openmp<16"
cd src/croco/models/curope
python setup.py build_ext --inplace
cd ../../../..
```

Dependencies are not fully version-locked; see the release checklist.

## Checkpoint

Obtain `cut3r_512_dpt_4_64.pth` from the
[official CUT3R checkpoint instructions](https://github.com/CUT3R/CUT3R#download-checkpoints)
and place it under `src/`, or pass an external path with `--model_path` for
inference and `--weights` for evaluation. This is an upstream starting checkpoint.
Exact BAT3R experiment checkpoint provenance is still needed for reproducing
reported results; see the release checklist.

## Inference

```bash
CUDA_VISIBLE_DEVICES=0 python bat3r_demo.py \
    --model_path src/cut3r_512_dpt_4_64.pth \
    --seq_path examples/westlake.mp4 \
    --size 512 --output_dir tmp/bat3r \
    --model_update_type bat3r
```

The demo writes per-frame artifacts and a merged `video_merged.ply` with camera
metadata. `--seq_path` accepts a video or image directory.

| Argument | Default | Purpose |
| --- | --- | --- |
| `--model_path` | `src/cut3r_512_dpt_4_64.pth` | Checkpoint path |
| `--model_update_type` | `bat3r` | `bat3r`, legacy `ttt3r`, or reference `cut3r` |
| `--size` | `512` | Input resolution |
| `--frame_interval` | `1` | Frame sampling interval |
| `--reset_interval` | `1000000` | State reset interval |
| `--vis_threshold` | `1.5` | Point confidence threshold |

`bash bat3r_run.sh ...` forwards arguments to the demo. Set `PYTHON` to use
an existing Python environment. The old `demo.py` and `run.sh` names still work.

## Static reconstruction renders

```bash
python bat3r_reconstruct.py \
    --model_path src/cut3r_512_dpt_4_64.pth \
    --seq_path examples/westlake.mp4 \
    --model_update_type bat3r --output_dir tmp/bat3r_render \
    --render_dir tmp/bat3r_render/png --render_views 8 \
    --render_engine matplotlib
```

Use `filament` or `glfw` for Open3D rendering when available. See
`python bat3r_reconstruct.py --help` for rendering options.

## Evaluation

```bash
python bat3r_eval.py --task relpose \
    --weights src/cut3r_512_dpt_4_64.pth \
    --eval_dataset tum_s1_90 --size 512
```

The launcher defaults to `--model_update_type bat3r` and stores outputs under
`eval_results/bat3r/<task>`. Use a separate `--output_dir` for every dataset,
configuration, and seed to preserve results. Add `--dry-run` to inspect the
forwarded arguments without model execution.

See [evaluation instructions](eval/README.md) for pose, depth, reconstruction,
data preparation, and benchmark-specific flags.

The historical `ttt3r` mode is an alias for modified BAT3R paths in this
repository. It is not an independent upstream TTT3R baseline. The `cut3r` mode
selects the existing reference update path in this modified model.

## Validation

```bash
python -m unittest discover -s tests -v
```

CPU tests check compatibility imports, state-dict loading, confidence layouts,
and launcher dispatch. Optional exact numerical parity uses an ignored copy of
the pre-migration Bayesian module at `tests/_baseline_bayesian_ttt.py`.
GitHub Actions checks Python syntax and portable launcher syntax without a GPU.

## Acknowledgements and license notices

Inherited components retain their original names and notices:
[TTT3R](https://github.com/Inception3D/TTT3R),
[CUT3R](https://github.com/CUT3R/CUT3R),
[DUSt3R](https://github.com/naver/dust3r),
[MonST3R](https://github.com/Junyi42/monst3r), and
[CroCo](https://github.com/naver/croco).

See [LICENSE](LICENSE) and the bundled third-party notices, including
[src/croco/LICENSE](src/croco/LICENSE). The root file preserves upstream CUT3R
CC BY-NC-SA 4.0 and TTT3R MIT notices; no Apache-2.0 project-wide grant is asserted.
Paper and citation metadata will be added when confirmed.
