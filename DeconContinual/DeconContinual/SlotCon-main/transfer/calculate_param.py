import torch

checkpoint_path = "/work/vision_lab/DenseSSL/FPN/SlotCon-main/output/slotcon/current.pth"
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# If the checkpoint was saved with something like:
# torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)
# then you likely need to do:
if "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
    print("y")
elif "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
else:
    # Otherwise, assume the checkpoint itself is the state_dict
    state_dict = checkpoint
#print(state_dict)
# Sum the number of elements in each tensor
# print(state_dict["model"].keys)
# total_params = sum(t.numel() for t in state_dict["model"].values() if hasattr(t, "numel"))
# print(f"Total parameters in checkpoint: {total_params}")
#dict_keys(['args', 'model', 'optimizer', 'scheduler', 'epoch', 'scaler'])

model_state_dict = checkpoint["model"]

s = 0
# Iterate through each parameter tensor in the state dictionary
for name, param in model_state_dict.items():
    # Make sure we're dealing with a tensor
    if hasattr(param, "numel"):
        print(f"{name} -> Shape: {tuple(param.shape)}, Params: {param.numel()}")
        s= s + param.numel()
total_params = sum(t.numel() for t in model_state_dict.values() if hasattr(t, "numel"))
print(s)

print(total_params)
