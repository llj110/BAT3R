# Evaluation

## Datasets
Please follow [MonST3R](https://github.com/Junyi42/monst3r/blob/main/data/evaluation_script.md) and [Spann3R](https://github.com/HengyiWang/spann3r/blob/main/docs/data_preprocess.md) to download **ScanNet**, **TUM-dynamics**,**Sintel**, **Bonn**, **KITTI**   and **7scenes** datasets.

### ScanNet
To prepare the **ScanNet** dataset, execute:
```bash
python datasets_preprocess/long_prepare_scannet.py # You may need to change the path of the dataset
```

### TUM-dynamics
To prepare the **TUM-dynamics** dataset, execute:
```bash
python datasets_preprocess/long_prepare_tum.py # You may need to change the path of the dataset
```

### Bonn
To prepare the **Bonn** dataset, execute:
```bash
python datasets_preprocess/long_prepare_bonn.py # You may need to change the path of the dataset
```

### KITTI
To prepare the **KITTI** dataset, execute:
```bash
python datasets_preprocess/long_prepare_kitti.py # You may need to change the path of the dataset
```

# Evaluation Scripts

Results will be saved in `eval_results/*`.

### Camera Pose Estimation

```bash
CUDA_VISIBLE_DEVICES=6,7 bash eval/relpose/run_scannet.sh # You may need to change [--num_processes] to the number of your gpus and choose sequence length in datasets=('scannet_s3_1000')
CUDA_VISIBLE_DEVICES=6,7 bash eval/relpose/run_tum.sh # You may need to change [--num_processes] to the number of your gpus and choose sequence length in datasets=('tum_s1_1000')
CUDA_VISIBLE_DEVICES=6,7 bash eval/relpose/run_sintel.sh # You may need to change [--num_processes] to the number of your gpus
```

### Video Depth

```bash
CUDA_VISIBLE_DEVICES=5 bash eval/video_depth/run_kitti.sh # You may need to change [--num_processes] to the number of your gpus and choose sequence length in datasets=('kitti_s1_500')
CUDA_VISIBLE_DEVICES=5 bash eval/video_depth/run_bonn.sh # You may need to change [--num_processes] to the number of your gpus and choose sequence length in datasets=('bonn_s1_500')
CUDA_VISIBLE_DEVICES=5 bash eval/video_depth/run_sintel.sh # You may need to change [--num_processes] to the number of your gpus
```



### 3D Reconstruction

```bash
CUDA_VISIBLE_DEVICES=5 bash eval/mv_recon/run.sh # You may need to change [--num_processes] to the number of your gpus and hoose sequence length in max_frames
```

