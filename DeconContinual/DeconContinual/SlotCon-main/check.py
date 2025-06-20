import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch

# Step 1: Simulate a random RGB input image (224x224)
image = np.random.rand(224, 224, 3)
image = (image - image.min()) / (image.max() - image.min())  # Normalize

# Step 2: Simulate a feature map output from SlotCon layer3 [C=512, h=28, w=28]
feature_map = torch.rand(512, 28, 28)  # Fake activation map

# Step 3: Average over channels to get a [28x28] activation heatmap
heatmap = torch.mean(feature_map, dim=0).numpy()
heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())  # Normalize

# Step 4: Resize heatmap to match original image size
heatmap_resized = cv2.resize(heatmap, (224, 224))

# Step 5: Overlay heatmap on original image
import matplotlib.pyplot as plt

# Create figure and axes
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# --- Original Image ---
axes[0].imshow(image)
axes[0].set_title("Original Image")
axes[0].axis('off')

# --- Activation Heatmap ---
im = axes[1].imshow(heatmap_resized, cmap='jet')
axes[1].set_title("Activation Heatmap")
axes[1].axis('off')

# --- Overlay ---
axes[2].imshow(image, alpha=0.6)
axes[2].imshow(heatmap_resized, cmap='jet', alpha=0.4)
axes[2].set_title("Overlay")
axes[2].axis('off')

# Adjust layout and save
plt.tight_layout()
plt.savefig("slotcon_mock_featuremap_overlay.png", dpi=300, bbox_inches='tight')
plt.show()