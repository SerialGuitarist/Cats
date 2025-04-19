import os

import gzip
import numpy as np
import matplotlib.pyplot as plt

def load_mnist():
    filenames = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz',
             't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']
    filenames = ["data/mnist/" + filename for filename in filenames]
    data = []
    for filename in filenames:
        with gzip.open(filename, 'rb') as f:
            if 'labels' in filename:
                ## load the labels as a one-dimensional array of integers
                data.append(np.frombuffer(f.read(), np.uint8, offset=8))
            else:
                ## load the images as a two-dimensional array of pixels
                data.append(np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,1,28,28))
    
    return data

def save_images(images, labels=None, save_path="out.png", title=""):
    ## images expected to be [n, 1, 28, 28]
    ## remove the channel dimension
    pixels = images.squeeze(1)
    num_images = len(images)

    fig_width = max(10, 2 * num_images)
    fig_height = 3
    fig, axs = plt.subplots(ncols=num_images, nrows=1, figsize=(fig_width, fig_height))

    ## ensure axs is always iterable
    if num_images == 1:
        axs = [axs]

    ## loop over the images and display them with their labels
    for i in range(len(images)):
        ## display the image and its labe\w:bl
        axs[i].imsh:e ow(pixels[i], cmap="gray")
        axs[i].set_title(labels[i] if labels is not None else "")
        ## remove the tick marks and axis labels
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].set_xlabel(f"Index: {i}")

    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])  ## leave room for subtitle

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved image grid to: {save_path}")

