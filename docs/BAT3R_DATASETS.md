# BAT3R dataset resources

Download data from its original provider, then prepare the layout expected by
the selected BAT3R evaluator. Original downloads and prepared benchmark subsets
are different artifacts; preprocessing, sampling, and sequence lists determine
the actual evaluation protocol.

| Dataset | Official source / download | BAT3R preparation and configuration |
| --- | --- | --- |
| 7-Scenes | [Microsoft Research and scene downloads](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/) | [Multi-view loader](../eval/mv_recon/data.py); root `data/7scenes`. The loader expects projected depth files named `frame-XXXXXX.depth.proj.png`. See [SimpleRecon preprocessing](https://github.com/nianticlabs/simplerecon/blob/main/data_scripts/7scenes_preprocessing.py) and [Spann3R notes](https://github.com/HengyiWang/spann3r/blob/main/docs/data_preprocess.md#7-scenes). |
| TUM RGB-D | [Dataset downloads](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download) · [Format](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats) · [Evaluation tools](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/tools) | [Long-sequence preparation](../datasets_preprocess/long_prepare_tum.py) · [Pose metadata](../eval/relpose/metadata.py) |
| ScanNet | [Official project repository](https://github.com/ScanNet/ScanNet) · [Official download/access instructions](https://github.com/ScanNet/ScanNet#scannet-data) · [RGB-D exporter](https://github.com/ScanNet/ScanNet/tree/master/SensReader/python) | [Long-sequence preparation](../datasets_preprocess/long_prepare_scannet.py) · [Pose metadata](../eval/relpose/metadata.py) · [PLY data paths](../eval/mv_recon/bat3r_data_paths.py) |
| KITTI | [Odometry benchmark](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) · [Depth prediction benchmark](https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction) | [Long-sequence preparation](../datasets_preprocess/long_prepare_kitti.py) · [Depth metadata](../eval/video_depth/metadata.py) · [PLY data paths](../eval/mv_recon/bat3r_data_paths.py) |
| Bonn RGB-D Dynamic | [Dataset and sequence downloads](https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/) | [Long-sequence preparation](../datasets_preprocess/long_prepare_bonn.py) · [Depth metadata](../eval/video_depth/metadata.py) |
| Sintel | [MPI Sintel benchmark](http://sintel.is.tue.mpg.de/) · [MonST3R download/preparation recipe](https://github.com/Junyi42/monst3r/blob/main/data/evaluation_script.md#sintel) | [Depth metadata](../eval/video_depth/metadata.py) · [Pose metadata](../eval/relpose/metadata.py) |
| Neural RGB-D (`NRGBD`) | [Original project](https://dazinovic.github.io/neural-rgbd-surface-reconstruction/) · [Dataset ZIP](https://kaldir.vc.in.tum.de/neural_rgbd/neural_rgbd_data.zip) · [Spann3R preparation notes](https://github.com/HengyiWang/spann3r/blob/main/docs/data_preprocess.md#neural-rgbd) | [Multi-view loader](../eval/mv_recon/data.py); BAT3R root `data/nrgbd` |

The legacy preprocessing scripts contain local paths that must be set for your
installation. Inspect the linked metadata before choosing an `--eval_dataset`
variant such as `tum_s1_90` or `bonn_s1_110`. A download alone does not create
those prepared subsets.

For evaluation commands, return to [the evaluation guide](../eval/README.md).
For inherited preparation examples, see [MonST3R](https://github.com/Junyi42/monst3r/blob/main/data/evaluation_script.md)
and [Spann3R](https://github.com/HengyiWang/spann3r/blob/main/docs/data_preprocess.md).
Sintel's legacy HTTP site can be intermittently unavailable; the MonST3R recipe
provides the corresponding source and preparation commands.
