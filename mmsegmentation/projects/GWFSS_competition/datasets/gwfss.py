# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS

@DATASETS.register_module()
class GWFSSDataset(BaseSegDataset):
    """Custom dataset for GWFSS (Global Wheat Field Semantic Segmentation).

    Each pixel in the segmentation map corresponds to one of:
    0 - background, 1 - head, 2 - stem, 3 - leaf.
    """

    METAINFO = dict(
        classes=('background', 'head', 'stem', 'leaf'),
        palette=[[0, 0, 0],        # background
                 [132, 255, 50],   # head
                 [255, 132, 50],   # stem
                 [50, 255, 214]]   # leaf
    )

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            **kwargs)