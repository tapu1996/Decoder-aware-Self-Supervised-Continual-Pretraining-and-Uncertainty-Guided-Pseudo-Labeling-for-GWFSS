#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import sys
import torch

def convert_beit_to_mmseg(beit_ckpt):
    new_state_dict = {}
    for k, v in beit_ckpt.items():
        # original_k = k

        # if not k.startswith("blocks."):
        #     continue  # Skip irrelevant keys

        # # Convert prefix
        # k = k.replace("blocks.", "layers.")

        # # Rename submodules
        # k = k.replace("mlp.fc1.", "ffn.layers.0.0.")
        # k = k.replace("mlp.fc2.", "ffn.layers.1.")
        # k = k.replace("norm1", "ln1")
        # k = k.replace("norm2", "ln2")

        # # Add MMSeg-style prefix
        # k = "backbone." + k

        new_state_dict[k] = v
        #print(f"{original_k} → {k}")

    return new_state_dict


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_beit_ckpt.py <input_ckpt.pth> <output_ckpt.pth>")
        sys.exit(1)

    input_ckpt = sys.argv[1]
    output_ckpt = sys.argv[2]

    # Load checkpoint
    obj = torch.load(input_ckpt, map_location="cpu")
    if "module" in obj:
        obj = obj["module"]
    else:
        raise Exception("Expected 'module' key in checkpoint")

    # Convert BEiT -> MMSeg style
    converted_state_dict = convert_beit_to_mmseg(obj)

    # Save new checkpoint
    torch.save({"state_dict": converted_state_dict}, output_ckpt)
    print(f"\n✅ Converted checkpoint saved to: {output_ckpt}")
