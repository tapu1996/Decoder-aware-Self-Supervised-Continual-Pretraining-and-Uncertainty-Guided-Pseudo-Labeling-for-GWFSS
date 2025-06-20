# GWFSS_Challenge

#Pretraining


Environment installation

virtualenv --no-download densesslenv
source densesslenv/bin/activate

pip install --upgrade pip
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install termcolor
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2
pip install timm

Download the weight from https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_384.pth. Rename it to current.pth and copy it to output-dir first.



cd DeconContinual/DeconContinual

torchrun --master_port 12348 --nproc_per_node=8 SlotCon-main/main_pretrain_continual.py --dataset gwfss --data-dir /home/ubuntu/DenseSSL/Data/gwfss/gwfss_competition_pretrain --output-dir /home/ubuntu/DenseSSL/output/convnext-l --arch convnext_large --dim-hidden 4096 --dim-out 256 --num-prototypes 256 --teacher-momentum 0.99 --teacher-temp 0.07 --group-loss-weight 0.5  --group-loss-weight-dec 0.5 --encoder-loss-weight 0 --batch-size 224 --optimizer adamW --base-lr 0.0002 --weight-decay 0.05 --warmup-epoch 5 --epochs 50 --fp16 --print-freq 10 --save-freq 1 --auto-resume --num-workers 12 --use-decoder --decoder-type FPN --sk-channel-dropout-prob 0.5 -dds --resume-new --copy-encoder --load-otherweightsThanSlotcon --freeze-encoder


torchrun --master_port 12348 --nproc_per_node=8 SlotCon-main/main_pretrain_continual.py --dataset gwfss --data-dir /home/ubuntu/DenseSSL/Data/gwfss/gwfss_competition_pretrain --output-dir /home/ubuntu/DenseSSL/output/convnext-l --arch convnext_large --dim-hidden 4096 --dim-out 256 --num-prototypes 256 --teacher-momentum 0.99 --teacher-temp 0.07 --group-loss-weight 0.5  --group-loss-weight-dec 0.5 --encoder-loss-weight 0 --batch-size 224 --optimizer adamW --base-lr 0.0002 --weight-decay 0.05 --warmup-epoch 5 --epochs 250 --fp16 --print-freq 10 --save-freq 1 --auto-resume --num-workers 12 --use-decoder --decoder-type FPN --sk-channel-dropout-prob 0.5 -dds --resume-new
