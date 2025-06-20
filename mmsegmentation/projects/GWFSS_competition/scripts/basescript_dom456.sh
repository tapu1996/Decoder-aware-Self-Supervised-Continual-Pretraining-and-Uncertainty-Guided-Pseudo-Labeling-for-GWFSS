#!/bin/bash
#SBATCH --account=def-farma
#SBATCH --time=12:0:0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100GB
#SBATCH --gres=gpu:a100_3g.20gb:1
source /home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/openmmlab/bin/activate
cd /home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition
#python /home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/tools/train.py /home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/Fundus/refuge/configs/fcn_d6_r50-d16_513x513_30k_voc12aug_moco_BN.py --work-dir /home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/Fundus/refuge/work_dirs/convnext_slotcon_base3_1

#python /home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/tools/train.py /home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/configs/ConvNextUpperNet_base.py --work-dir /home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/work_dirs/ConvNextImageNet_base_vicregL_1K


#python /home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/tools/train.py /home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/configs/ConvNextUpperNet_large_512.py --work-dir /home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/work_dirs/Slotcon-continual-higher300ep-highlrfinetune_corrected
#python /home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/utils/similarity_measure.py --domain 4
python /home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/utils/similarity_measure.py --domain 3

python /home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/utils/similarity_measure.py --domain 5
#python /home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/utils/similarity_measure.py --domain 6
#python /home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/tools/train.py /home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/Fundus/refuge/configs/similarasBaseVoc.py --work-dir /home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/Fundus/refuge/work_dirs/DeconMLConvnextFCN