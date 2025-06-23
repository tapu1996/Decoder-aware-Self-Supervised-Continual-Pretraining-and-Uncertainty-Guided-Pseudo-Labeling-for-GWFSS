For semantic segmentation tasks we have used mmsegmentation

The project folder is inside mmsegmentation/projects/GWFSS_competition

All the config files are available inside mmsegmentation/projects/GWFSS_competition/configs

***How did I train***

We used the following config files for training the final models

For Training from pseudo label (ConvNext-L)
---Initialized ImageNet1K: ConvNextUpperNet_large_512.py
---Initialized GWFSS pretrained backbone: ConvNextUpperNet_large_512Decon

We used the pseudo label trained weight to finally train on 5 folds (5x2 = 10 models)
  
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

Configs for Beit training
---beit_large_512f0
---beit_large_512f1
---beit_large_512f2
---beit_large_512f3
---beit_large_512f4


to train the model please check inside the dataloader and change the data paths of data_root, also check inside train_dataloader and val_dataloader. Make sure that your checkpoint path is correct as well.

Training command:

cd mmsegmentation
python tools/train.py [Config File Path] --work-dir [Path to save log and weights]


Inference 
Before doing inference please change all the config paths, checkpoints, data and output dir. ConvNextUpperNet_large_768.py config was used for ConvNeXt models, beit_large_512f0.py was used for BEiT models. We have ensembled all the prediction.
We have used absolute path...please replace it with your absolute path

cd mmsegmentation
python utils\Inference.py


Pseudo Label Generation
We have used a 2-stage pseudo label policy.
---Code used for Similarity measure:  mmsegmentation/projects/GWFSS_competition/utils/similarity_measure.py
---Code used for Uncertainity Estimation measure:  mmsegmentation/projects/GWFSS_competition/utils\script_10fold_uncertainty.py


