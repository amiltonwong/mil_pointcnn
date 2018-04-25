#!/usr/bin/env bash

gpu=
setting=
modelsFolder="../../models/seg/"

usage() { echo "train/val pointcnn_seg with -g gpu_id -x setting options"; }

gpu_flag=0
setting_flag=0
while getopts g:x:h opt; do
  case $opt in
  g)
    gpu_flag=1;
    gpu=$(($OPTARG))
    ;;
  x)
    setting_flag=1;
    setting=${OPTARG}
    ;;
  h)
    usage; exit;;
  esac
done

shift $((OPTIND-1))

if [ $gpu_flag -eq 0 ]
then
  echo "-g option is not presented!"
  usage; exit;
fi

if [ $setting_flag -eq 0 ]
then
  echo "-x option is not presented!"
  usage; exit;
fi

if [ ! -d "$modelsFolder" ]
then
  mkdir -p "$modelsFolder"
fi

echo "Train/Val with setting $setting on GPU $gpu!"
CUDA_VISIBLE_DEVICES=$gpu python3 ../train_val_seg.py -t ../../data/scannet/scannet_split_dataset/train_files.txt -v ../../data/scannet/scannet_split_dataset/val_files.txt -s ../models/seg -m pointcnn_seg -x $setting > ../../models/seg/pointcnn_seg_$setting.txt 2>&1 &
