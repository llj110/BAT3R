#!/bin/bash

set -e

workdir='.'
model_names=('ttt3r') # ttt3r cut3r

ckpt_name='cut3r_512_dpt_4_64'
model_weights="${workdir}/src/${ckpt_name}.pth"

# datasets=('scannet_s3_200' 'scannet_s3_90' 'scannet_s3_100' 'scannet_s3_150' 'scannet_s3_200' 'scannet_s3_300' 'scannet_s3_400' 'scannet_s3_500'
#                'scannet_s3_600' 'scannet_s3_700' 'scannet_s3_800' 'scannet_s3_900' 'scannet_s3_1000')
datasets=('scannet_s3_90')

for model_name in "${model_names[@]}"; do
for data in "${datasets[@]}"; do
    output_dir="${workdir}/eval_results/relpose/${data}/${model_name}"
    echo "$output_dir"
    accelerate launch --num_processes 1 --main_process_port 29550 eval/relpose/launch.py \
        --weights "$model_weights" \
        --output_dir "$output_dir" \
        --eval_dataset "$data" \
        --size 512 \
        --model_update_type "$model_name"
done
done


