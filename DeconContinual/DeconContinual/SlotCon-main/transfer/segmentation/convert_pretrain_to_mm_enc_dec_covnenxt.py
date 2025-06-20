#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import sys
import torch

if __name__ == "__main__":
    input = sys.argv[1]
    print(input)

    obj = torch.load(input, map_location="cpu")
    if "state_dict" in obj:
        obj = obj["state_dict"]
    elif "model" in obj:
        obj = obj["model"]
    else:
        raise Exception("Cannot find the model in the checkpoint.")
    
    new_model = {}
    m = ""
    for k, v in obj.items():
        if k.startswith(f"module."):
            m= "module."
    for k, v in obj.items():
        # print(k)
        if not k.startswith(f"{m}encoder_k.") or "fc" in k:
            continue
        old_k = k
        if k.startswith(f"{m}encoder_k."):
            k = k.replace(f"{m}encoder_k.", "")
        k = k.replace("pwconv", "pointwise_conv")
        k = k.replace("dwconv", "depthwise_conv")
        print(old_k, "->", k)
        new_model[k] = v
    
    res = {"state_dict": new_model}
    torch.save(res, sys.argv[2])
    
    new_model = {}
    for k, v in obj.items():
        #print(k)
        if not k.startswith(f"{m}decoder_k.") or "fc" in k:
            continue
        old_k = k
        if k.startswith(f"{m}decoder_k."):
            k = k.replace(f"{m}decoder_k.", "")
        print(old_k, "->", k)
        new_model[k] = v
    
    res = {"state_dict": new_model}

    torch.save(res, sys.argv[3])
