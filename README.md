# Decoder-aware Self-Supervised Continual Pretraining and Uncertainty-Guided Pseudo-Labeling for Wheat Organ Segmentation
:trophy: Winner of innovation award of Global wheat full semantic segmenation challenge.

:white_check_mark: This method has been accepted in CVPPA workshop (ICCV 2025)
# Pretraining


## Environment installation
```
virtualenv --no-download densesslenv
source densesslenv/bin/activate

pip install --upgrade pip
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install termcolor
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2
pip install timm
```


Download the weight from https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_384.pth. Rename it to current.pth and copy it to output-dir first.

```
cd DeconContinual/DeconContinual

torchrun --master_port 12348 --nproc_per_node=8 SlotCon-main/main_pretrain_continual.py --dataset gwfss --data-dir /home/ubuntu/DenseSSL/Data/gwfss/gwfss_competition_pretrain --output-dir /home/ubuntu/DenseSSL/output/convnext-l --arch convnext_large --dim-hidden 4096 --dim-out 256 --num-prototypes 256 --teacher-momentum 0.99 --teacher-temp 0.07 --group-loss-weight 0.5  --group-loss-weight-dec 0.5 --encoder-loss-weight 0 --batch-size 224 --optimizer adamW --base-lr 0.0002 --weight-decay 0.05 --warmup-epoch 5 --epochs 50 --fp16 --print-freq 10 --save-freq 1 --auto-resume --num-workers 12 --use-decoder --decoder-type FPN --sk-channel-dropout-prob 0.5 -dds --resume-new --copy-encoder --load-otherweightsThanSlotcon --freeze-encoder


torchrun --master_port 12348 --nproc_per_node=8 SlotCon-main/main_pretrain_continual.py --dataset gwfss --data-dir /home/ubuntu/DenseSSL/Data/gwfss/gwfss_competition_pretrain --output-dir /home/ubuntu/DenseSSL/output/convnext-l --arch convnext_large --dim-hidden 4096 --dim-out 256 --num-prototypes 256 --teacher-momentum 0.99 --teacher-temp 0.07 --group-loss-weight 0.5  --group-loss-weight-dec 0.5 --encoder-loss-weight 0 --batch-size 224 --optimizer adamW --base-lr 0.0002 --weight-decay 0.05 --warmup-epoch 5 --epochs 250 --fp16 --print-freq 10 --save-freq 1 --auto-resume --num-workers 12 --use-decoder --decoder-type FPN --sk-channel-dropout-prob 0.5 -dds --resume-new
```

# Segmentation
We have used mmsegmentation pipeline for this. 

## Environment Setup
Make sure to install it in editable format 
please check mmseg_requirements.txt if you have issue with versions.
```
cd mmsegmentation
pip install -e .
```


The project folder is inside mmsegmentation/projects/GWFSS_competition

All the config files are available inside mmsegmentation/projects/GWFSS_competition/configs

## Training
We used the following config files for training the final models

For Training from pseudo label (ConvNext-L)
---Initialized ImageNet1K: ConvNextUpperNet_large_512.py
---Initialized GWFSS pretrained backbone: ConvNextUpperNet_large_512Decon

We used the pseudo label trained weight to finally train on 5 folds (5x2 = 10 models)
```  
---ConvNextUpperNet_large_512f0.py
---ConvNextUpperNet_large_512f1.py
---ConvNextUpperNet_large_512f2.py
---ConvNextUpperNet_large_512f3.py
---ConvNextUpperNet_large_512f4.py
---ConvNextUpperNet_large_512f0_c
---ConvNextUpperNet_large_512f1_c
---ConvNextUpperNet_large_512f2_c
---ConvNextUpperNet_large_512f3_c
---ConvNextUpperNet_large_512f4_c
```
Configs for Beit training
```
---beit_large_512f0
---beit_large_512f1
---beit_large_512f2
---beit_large_512f3
---beit_large_512f4
```

to train the model please check inside the dataloader and change the data paths of data_root, also check inside train_dataloader and val_dataloader. Make sure that your checkpoint path is correct as well.

Training command:
```
cd mmsegmentation
python tools/train.py [Config File Path] --work-dir [Path to save log and weights]
```

## Inference 
Before doing inference please change all the config paths, checkpoints, data and output dir. ConvNextUpperNet_large_768.py config was used for ConvNeXt models, beit_large_512f0.py was used for BEiT models. We have ensembled all the prediction.
We have used absolute path. Please replace it with your absolute path
```
cd mmsegmentation/projects/GWFSS_competition
python utils/Inference.py
```

## Pseudo Label Generation
We have used a 2-stage pseudo label policy.

---Code used for Similarity measure:  mmsegmentation/projects/GWFSS_competition/utils/similarity_measure.py

---Code used for Uncertainity Estimation measure:  mmsegmentation/projects/GWFSS_competition/utils/script_10fold_uncertainty.py


# Model Weights
Link: 
```
https://uofc-my.sharepoint.com/:f:/g/personal/tapotosh_ghosh_ucalgary_ca/Et5pTPRcF0tGjyZPa32_QmQBZJPaII1c5hq7uV6EbNC29g?e=Xf3geN
```

Folder - ModelsUsedForPseudoGeneration+PretrainedCheckpoints.

1. Pretrained DeConML - The pretrained ConvNext-L model was trained using a two-stage continual DeConML approach with an FPN (Feature Pyramid Network) decoder. In the first phase, the encoder was initialized with ImageNet-1K weights (available in https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_384.pth), and its layers were frozen for the first 50 epochs. During this phase, only the decoder and other randomly initialized layers were pretrained on the given unlabelled pretraining dataset. In the second phase, the entire encoder-decoder block was fine-tuned on the same dataset, without any frozen layers. This pretrained ConvNext-L backbone was then used to find similar images based on a given training set and to initialize the ConvNext-L model during training with labeled data.

2. ConvnextLabelledDataonly and DeConMLLabelledDataOnly: Encoders in the ConvNextLabelledDataOnly folder are initialized with the ConvNext-L ImageNet-1K pretrained model, while encoders in the DeConMLLabelledDataOnly folder are initialized with the pretrained ConvNext-L from the unlabelled pretraining step (available in the Pretrained DeConML folder). Both sets of models were trained using 5-fold cross-validation. These pretrained models were then utilized for the second stage of pseudo-labelling.

3. Trained on Pseudo Only: Here, the model encoders have been initialized with ConvNext-L ImageNet-1K weight and continual 2-stage DeConML pretrained weights. These models are trained only on pseudo data only.


Folder - Final_models (Use it for inference)

4. Final Ensemble Models:
   
	4.1 DeconPseudoFirstMainSecond: The full model is initialized with pseudolabelled weight (3. Trained on Pseudo Only - ConvNext-L DeConML weight) and trained with only the given 99 labelled dataset with a lower learning rate compared to stage 3: Trained on Pseudo Only.
   
	4.2 ConvNextPseudoFirstMainSecond: The full model is initialized with pseudolabelled weight (3. Trained on Pseudo Only - ConvNext-L ImageNet-1k weight) and trained with only the given 99 labelled dataset with a lower learning rate compared to stage 3: Trained on Pseudo Only.
   
	4.3 beitFolds: It contained models trained with only the given 99 labelled dataset. The encoder was initialized with BeitV2 ImageNet-1K weight beitv2_large_patch16_224_pt1k_ft1k (available on: https://github.com/microsoft/unilm/tree/master/beit2)
	
All these weights available on 4.1, 4.2, 4.3 are ensembled by summing their logits to get the final result. Use the models inside Final_models for the inference.
