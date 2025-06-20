import os
import time
import glob
import torch
import torchvision
from PIL import Image
import tqdm
import sys

sys.path.append("/home/sebquet/scratch/VisionResearchLab/DenseSSL/DenseSSL/SlotCon-main")
from data.datasets import ImageFolder
from data.transforms import CustomDataAugmentation

root_path = "/localscratch/sebquet.34349208.0/data/COCO" # "/home/sebquet/scratch/VisionResearchLab/DenseSSL/Data/COCO"
data_path = os.path.join(root_path, "train2017")

transform = CustomDataAugmentation(size=224, min_scale=0.08)
train_dataset = ImageFolder("COCO", root_path, transform)

# for idx in tqdm.tqdm(range(len(train_dataset.fnames)), total=len(train_dataset.fnames)):
#     _ = train_dataset.__getitem__(idx)
# print()

# exit()

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=256,
    shuffle=True,
    num_workers=8,  # args.num_workers,
    pin_memory=False,
    sampler=None,
    drop_last=True,
    # prefetch_factor=None,
)
train_len = len(train_loader)
for i, batch in tqdm.tqdm(
    enumerate(train_loader), total=train_len, desc="loading with dataloader torch"
):
    crops, coords, flags = batch

exit(0)

files = glob.glob(os.path.join(data_path, "*.jpg"))
t0 = time.time()
for file in tqdm.tqdm(files, desc="Loading images", total=len(files)):
    image = Image.open(file).convert("RGB")
    # img = torchvision.io.decode_image(file)
print("it took", time.time() - t0, "to load all images")
