#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import pickle
import argparse
import torch
import copy

def convert_pretrain_to_d2(
        full_model_path:str, new_model_path:str, only_encoder:bool=True, 
        create_empty_ckpt:bool=False, pretraining_algo:str="slotcon", arch:str="resnet50"):
    if pretraining_algo == "slotcon":
        encoder_path_kw = "encoder_k"
        decoder_path_kw = "decoder_k"
    elif pretraining_algo == "densecl":
        encoder_path_kw = "backbone"
        decoder_path_kw = "decoder"
    else:
        raise ValueError(f"Unknown pretraining algorithm: {pretraining_algo}")


    obj = torch.load(full_model_path, map_location="cpu")
    if "state_dict" in obj:
        obj = obj["state_dict"]
    elif "model" in obj:
        obj = obj["model"]
    else:
        raise Exception("Cannot find the model in the checkpoint.")

    # get keyword
    start_kw = []
    for k, v in obj.items():
        param_kw = k.split(".")
        if encoder_path_kw in param_kw:
            for i, kw in enumerate(param_kw):
                if kw == encoder_path_kw:
                    break
                start_kw.append(kw)
            break
    if len(start_kw) == 0:
        start_kw = ""
    else:
        start_kw = ".".join(start_kw) + "."

    new_model = {}
    
    if only_encoder:
        arch_parts = [encoder_path_kw]
    else:
        arch_parts = [encoder_path_kw, decoder_path_kw]

    if "resnet" in arch:
        if not create_empty_ckpt: 
            for arch_part in arch_parts:
                for k, v in obj.items():
                    if not k.startswith(f"{start_kw}{arch_part}.") or "fc" in k:
                        continue
                    old_k = copy.deepcopy(k)
                    k = k.replace(f"{start_kw}{arch_part}.", "")
                    if arch_part == encoder_path_kw:
                        if ("layer" not in k):
                            k = "stem." + k
                        for t in [1, 2, 3, 4]:
                            k = k.replace("layer{}".format(t), "res{}".format(t + 1))
                        for t in [1, 2, 3]:
                            k = k.replace("bn{}".format(t), "conv{}.norm".format(t))
                        k = k.replace("downsample.0", "shortcut")
                        k = k.replace("downsample.1", "shortcut.norm")
                    print(old_k, "->", k)
                    new_model[k] = v
    elif "convnext" in arch:
        print("Just copying the encoder")
        if not create_empty_ckpt: 
            for arch_part in arch_parts:
                for k, v in obj.items():
                    if not k.startswith(f"{start_kw}{arch_part}.") or "fc" in k:
                        continue
                    old_k = copy.deepcopy(k)
                    k = k.replace(f"{start_kw}{arch_part}.", "")
                    print(old_k, "->", k)
                    new_model[k] = v
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    res = {
        "model": new_model,
        "__author__": "lalala",
        "matching_heuristics": True}

    with open(new_model_path, "wb") as f:
        pickle.dump(res, f)

def main():
    parser = argparse.ArgumentParser(description="Convert a pretrained model to Detectron2 format.")
    parser.add_argument('-pm', '--full-pretrained-model-path', type=str, help='Path to the full pretrained model')
    parser.add_argument('-nm', '--new-model-path', type=str, help='Path to save the new model')
    parser.add_argument('-pa', '--pretraining-algo', type=str, default="slotcon", help='Pretraining algorithm slotcon|densecl')
    parser.add_argument('--use-decoder', action="store_true")
    parser.add_argument('--empty', action="store_true")
    parser.add_argument('--arch', type=str, default="resnet50", help='Architecture resnet50|convnext_small')
    args = parser.parse_args()
    assert not args.full_pretrained_model_path == args.new_model_path, "The new model path should be different from the old model path."

    convert_pretrain_to_d2(
        full_model_path=args.full_pretrained_model_path, 
        new_model_path=args.new_model_path, 
        only_encoder= (not args.use_decoder), 
        create_empty_ckpt=args.empty, 
        pretraining_algo=args.pretraining_algo,
        arch=args.arch
        )
    

if __name__ == "__main__":
    main()