# BAT3R code map

The BAT3R implementation is under `src/bat3r/`. Names under `dust3r` and `croco`
identify inherited infrastructure and are retained for imports and checkpoints.

| File | Role |
| --- | --- |
| `src/bat3r/bat3r_model.py` | Recurrent reconstruction model, global context, pose gate, and state-update integration |
| `src/bat3r/bat3r_update.py` | BayesianRecurrentUnit, adaptive process noise, uncertainty fusion, and UpdateController |
| `src/bat3r/bat3r_inference.py` | Inference and recurrent sequence execution |
| `bat3r_demo.py` | Video/image inference and merged PLY export |
| `bat3r_reconstruct.py` | Reconstruction and static PNG rendering |
| `bat3r_eval.py` | Common evaluation entry point for pose, depth, multi-view reconstruction, and PLY export |
| `bat3r_run.sh` | Demo launcher using the selected Python environment |

## Compatibility

`dust3r.model`, `dust3r.inference`, and `dust3r.bayesian_ttt` re-export the
canonical implementation. `demo.py`, `render_video_recon.py`, and `run.sh`
remain compatibility entry points. `BAT3R` and `BAT3RConfig` alias the existing
`ARCroco3DStereo` classes; tensor names and state-dict layouts are retained.
The camera metadata markers in PLY files retain the `TTT3R_` spelling for
existing readers.

Use `--model_update_type bat3r` in new experiments. The historical `ttt3r`
value selects the same modified code paths in this repository and is retained
for old scripts. It must not be reported as an independent upstream TTT3R
baseline. `cut3r` selects the existing reference update path in this modified
model; use the upstream repository for an independently reproduced baseline.
The three inference functions follow distinct existing execution paths; the
demo and pose/depth evaluators use `inference_recurrent_lighter`.

## Python imports

With `src` on `PYTHONPATH`:

```python
from bat3r import BAT3R, BAT3RConfig
from bat3r.bat3r_update import BayesianRecurrentUnit, UpdateController
from bat3r.bat3r_inference import inference_recurrent_lighter
```

Utilities, datasets, heads, and CroCo are inherited and retain their original
names and notices. Moving them would obscure their provenance and disrupt
checkpoint compatibility.
