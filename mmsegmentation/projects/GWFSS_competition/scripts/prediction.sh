#!/bin/bash
#SBATCH --account=def-farma
#SBATCH --time=12:0:0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:a100:1
source /home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/openmmlab/bin/activate
python /home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/validation_competition/script_evaluated.py
#python /home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/tools/train.py /home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/Fundus/refuge/configs/similarasBaseVoc.py --work-dir /home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/Fundus/refuge/work_dirs/DeconMLConvnextFCN