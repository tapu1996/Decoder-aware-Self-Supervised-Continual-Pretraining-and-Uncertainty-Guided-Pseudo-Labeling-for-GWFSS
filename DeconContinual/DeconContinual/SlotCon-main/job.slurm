#!/bin/bash
#SBATCH --partition=gpu-a100
#SBATCH --time=24:0:0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1

cd "/work/vision_lab/DenseSSL/SlotCon-main"
conda activate slotcon2
./scripts/slotcon_coco_r50_800ep_arc.sh 