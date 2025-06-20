import os
import sys

# Add the current directory to PYTHONPATH
current_directory = os.getcwd()

#sys.path.append(current_directory)
sys.path.append("/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition")

_base_ = [
    './gwfss_dataloaderf0Pseudo.py', 'models/beit_large.py',
    'mmseg::_base_/default_runtime.py',
    'schedulers/beit40k.py'
]
custom_imports = dict(imports='datasets.gwfss')
img_scale = (512, 512)
data_preprocessor = dict(size=img_scale)



checkpoint_path = "/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/model/beit_large_inkcheck.pth"
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(init_cfg=dict(type='Pretrained', checkpoint=checkpoint_path)),
    decode_head=dict(num_classes=4),
    auxiliary_head=dict(num_classes=4),
    test_cfg=dict(mode='whole', _delete_=True))
vis_backends = None
visualizer = dict(vis_backends=vis_backends)
