#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import torch
from convnext import add_convnext_config
from convnext import build_convnext_fpn_backbone

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator, PascalVOCDetectionEvaluator
from detectron2.layers import get_norm
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY, Res5ROIHeads


@ROI_HEADS_REGISTRY.register()
class Res5ROIHeadsExtraNorm(Res5ROIHeads):
    """
    As described in the MOCO paper, there is an extra BN layer
    following the res5 stage.
    """
    def _build_res5_block(self, cfg):
        seq, out_channels = super()._build_res5_block(cfg)
        norm = cfg.MODEL.RESNETS.NORM
        norm = get_norm(norm, out_channels)
        seq.add_module("norm", norm)
        return seq, out_channels


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        if "coco" in dataset_name:
            return COCOEvaluator(dataset_name, cfg, True, output_folder)
        else:
            assert "voc" in dataset_name
            return PascalVOCDetectionEvaluator(dataset_name)
    
    @classmethod
    def build_optimizer(cls, cfg, model):
        params = []
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        base_lr = cfg.SOLVER.BASE_LR

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            group_decay = 0.0 if "norm" in name or "bias" in name else weight_decay
            params.append({
                "params": [param],
                "weight_decay": group_decay
            })

        optim_name = cfg.SOLVER.OPTIMIZER.lower()
        if optim_name == "adamw":
            print(f"Using optimizer: {optim_name}")
            print(f"Total parameter groups: {len(params)}")
            return torch.optim.AdamW(params, lr=base_lr)
        elif optim_name == "sgd":
            return torch.optim.SGD(params, lr=base_lr, momentum=cfg.SOLVER.MOMENTUM)
        else:
            raise ValueError(f"Unsupported optimizer: {cfg.SOLVER.OPTIMIZER}")

def setup(args):
    cfg = get_cfg()
    # Loading default parmas for convnext
    add_convnext_config(cfg)
    # Ovewriting with config file
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
