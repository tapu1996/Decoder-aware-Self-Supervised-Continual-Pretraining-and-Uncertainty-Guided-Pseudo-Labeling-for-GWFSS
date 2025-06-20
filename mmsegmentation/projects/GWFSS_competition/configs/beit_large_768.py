import os
import sys

# Add the current directory to PYTHONPATH
current_directory = os.getcwd()

#sys.path.append(current_directory)
sys.path.append("/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition")

_base_ = [
    './gwfss_dataloader_larger.py', 'models/beit_large.py',
    'mmseg::_base_/default_runtime.py',
    'schedulers/beit40k.py'
]
custom_imports = dict(imports='datasets.gwfss')
img_scale = (768, 768)
data_preprocessor = dict(size=img_scale)





# model = dict(
#     data_preprocessor=data_preprocessor,
#     backbone=dict(init_cfg=dict(type='Pretrained', checkpoint="/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/Fundus/refuge/backbones/slotconBaseAdamW/enc_slotconBaseConvnext.pth")),
#     decode_head=dict(num_classes=21),
#     auxiliary_head=None)

checkpoint_path = "/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/model/beit_large_inkcheck.pth"
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(init_cfg=dict(type='Pretrained', checkpoint=checkpoint_path)),
    decode_head=dict(num_classes=4),
    auxiliary_head=dict(num_classes=4),
    test_cfg=dict(mode='whole', _delete_=True))

#    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(341, 341)))
vis_backends = None
visualizer = dict(vis_backends=vis_backends)
