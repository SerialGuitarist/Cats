import os

import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from PIL import Image

## give list of paths of specific images
## will load and return that as a torch tensor
def load_images(image_paths=[], image_size=64):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    images = []
    for path in image_paths:
        img = Image.open(path).convert("RGB") # 3 channel shenanigans
        img = transform(img)
        images.append(img)

    return torch.stack(images) # [n, 3, 64, 64]

## expects:
## data/cats/all/cat_0.png
## data/cats/all/cat_1.png
## data/cats/all/cat_2.png
## ...
def load_cats_dataset(data_dir="data/cats", image_size=64, batch_size=64, num_workers=2):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),  # converts to [0,1], shape [C,H,W]
    ])
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataset, loader

def save_images(images, labels=None, save_path="out.png", title=""):
    # [N, 3, H, W] -> [N, H, W, 3]
    pixels = images.permute(0, 2, 3, 1).cpu().numpy()
    num_images = len(images)

    ## create a figure with subplots for each image
    ## dynamically size figure: 3 inches tall, 3 inches wide per image
    fig_width = max(10, 2 * num_images)
    fig_height = 3
    fig, axs = plt.subplots(ncols=num_images, nrows=1, figsize=(fig_width, fig_height))

    ## ensuring axis is iterable
    if num_images == 1:
        axs = [axs]

    ## loop over images and disaply them with any associated labels
    for i in range(num_images):
        axs[i].imshow(pixels[i])
        axs[i].set_title(labels[i] if labels is not None else "")

        ## remove tick marks and axis labels
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        # axs[i].set_xlabel(f"Index: {i}")

    fig.suptitle(title, fontsize=16)
    ## leaving room for subtitle
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved image grid to: {save_path}")

if __name__ == "__main__":
    dataset, loader = load_cats_dataset("data/cats")
    images, labels = next(iter(loader))  ## images: [B, 3, 64, 64]

    ## let's take a look at the first 10
    save_images(images[:10], save_path="models/cats/cats_batch.png", title="Cats Batch")

