# 1. 设置数据集保存的目录路径
export dataset_dir="./data/7scenes"
mkdir -p $dataset_dir

# 2. 遍历7个场景并使用 wget 批量下载
for scene in chess fire heads office pumpkin redkitchen stairs; do
  echo "========== 正在下载场景: $scene =========="
  # wget -c http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/${scene}.zip -P $dataset_dir
  # 解压主压缩包
  unzip -q $dataset_dir/${scene}.zip -d $dataset_dir
  
  # 解压场景文件夹内的分卷序列压缩包 (seq-*.zip)
  unzip -q $dataset_dir/${scene}/'*.zip' -d $dataset_dir/${scene}
  
  # (可选) 删除原本的压缩包以节省空间
  rm $dataset_dir/${scene}.zip $dataset_dir/${scene}/*.zip
done