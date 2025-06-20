#!/bin/bash
#SBATCH --account=def-farma
#SBATCH --time=4:0:0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:a100:1
source /home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/openmmlab/bin/activate
cd /home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition
#python /home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/tools/train.py /home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/Fundus/refuge/configs/fcn_d6_r50-d16_513x513_30k_voc12aug_moco_BN.py --work-dir /home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/Fundus/refuge/work_dirs/convnext_slotcon_base3_1

#python  --work-dir /home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/work_dirs/ConvnextLPseudoLabelOnly
# python /home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/utils/script_10fold_uncertaintybeit.py
#python /home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/validation_competition/script_10fold_tta_cmd.py
#python /home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/tools/train.py /home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/configs/beit_large_512f4.py --work-dir /home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/work_dirs/beitFolds/beitfold4
#python /home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/tools/train.py /home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/configs/beit_large_512f0.py --work-dir /home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/work_dirs/BeitFolds2stageLower/fold0
python /home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/validation_competition/script10foldbeitensemble.py
#python /home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/tools/train.py /home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/Fundus/refuge/configs/similarasBaseVoc.py --work-dir /home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/Fundus/refuge/work_dirs/DeconMLConvnextFCN
#python /home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/validation_competition/script10foldbeitensemble.py
#python /home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/validation_competition/script_10fold_tta_cmd_beit.py
#python /home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/validation_competition/script_10fold_tta_cmd.py