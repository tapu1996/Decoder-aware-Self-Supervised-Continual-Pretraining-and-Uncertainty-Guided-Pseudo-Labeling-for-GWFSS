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
        parts = k.split(".", 1)  # split at the first dot only
        new_key = parts[1] if len(parts) > 1 else k  # remove first part if it exists
        new_model[new_key] = v
        # if k.startswith(f"norm0"):
        #     continue
        # if k.startswith(f"norm1"):
        #     continue
        # if k.startswith(f"norm2"):
        #     continue
        
        
        print(k,"--->",new_key)
        # new_model[k] = v
    
    res = {"state_dict": new_model}
    torch.save(res, sys.argv[2])
    