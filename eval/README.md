# BAT3R evaluation

Run all commands from the repository root. The common `bat3r_eval.py` launcher
accepts `--task relpose`, `video_depth`, `mv_recon`, or `ply` and forwards the
remaining options to the corresponding existing evaluator.

## Data preparation

See [official dataset downloads and BAT3R preparation links](../docs/BAT3R_DATASETS.md).
Download benchmark data from its official source and prepare the directory
layout expected by each evaluator. See [data preparation notes](eval.md),
`eval/relpose/metadata.py`, `eval/video_depth/metadata.py`, and
`eval/mv_recon/bat3r_data_paths.py`. Long-sequence preprocessing scripts under
`datasets_preprocess/` require setting dataset paths before use.

## Relative pose

```bash
python bat3r_eval.py --task relpose \
    --weights src/cut3r_512_dpt_4_64.pth \
    --eval_dataset tum_s1_90 --size 512 \
    --output_dir eval_results/bat3r/relpose/tum_s1_90
```

The pose evaluator uses `--eval_dataset`, not `--dataset`. TUM, ScanNet, Bonn,
KITTI, and Sintel variants are defined in its metadata; 7Scenes is handled by
multi-view reconstruction rather than this pose command.

## Video depth

```bash
python bat3r_eval.py --task video_depth \
    --weights src/cut3r_512_dpt_4_64.pth \
    --eval_dataset bonn_s1_110 --size 512 \
    --output_dir eval_results/bat3r/video_depth/bonn_s1_110
python eval/video_depth/eval_depth.py \
    --output_dir eval_results/bat3r/video_depth/bonn_s1_110 \
    --eval_dataset bonn_s1_110 --align metric
```

Depth scoring is a separate step. The existing scoring script also supports
`--align scale` and `--align "scale&shift"`; report the selected protocol.

## Multi-view reconstruction

```bash
python bat3r_eval.py --task mv_recon \
    --weights src/cut3r_512_dpt_4_64.pth \
    --dataset 7scenes --size 512 \
    --output_dir eval_results/bat3r/mv_recon/7scenes
```

The evaluator supports `7scenes`, `NRGBD`, and `all`. It writes PLY exports;
its general reconstruction accuracy/completion calculation is disabled in the
current implementation. Do not infer numerical benchmark scores from exports.

## KITTI / ScanNet PLY exports

```bash
python bat3r_eval.py --task ply \
    --weights src/cut3r_512_dpt_4_64.pth \
    --dataset scannet --max_frames 1000 \
    --output_dir eval_results/bat3r/ply/scannet
```

Use `--eval_metrics` on this specialized launcher to request its ICP and
reconstruction metrics. Data roots and sample lengths must match the prepared
layout in `bat3r_data_paths.py`.

## Mode names and historical scripts

- `bat3r`: public name for the existing BAT3R update paths.
- `ttt3r`: compatibility alias for the same modified paths, retained for existing
  experiments; it is not a faithful upstream TTT3R baseline.
- `cut3r`: the existing reference update path in this modified model.

`none` is not a supported mode. The original benchmark shell scripts remain
historical recipes and may contain server-specific Python paths and mode names;
prefer `bat3r_eval.py` for new runs. Record sequence lists, seeds, alignment,
sampling, and checkpoint hashes for every reported experiment.
