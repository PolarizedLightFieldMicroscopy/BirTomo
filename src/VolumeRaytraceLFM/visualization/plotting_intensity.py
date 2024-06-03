import matplotlib.pyplot as plt
import numpy as np


def plot_images(image_list):
    num_images = len(image_list)

    # Calculate the number of rows and columns for the subplots
    rows = int(np.sqrt(num_images))
    cols = int(np.ceil(num_images / rows))

    # Create a figure and subplots
    fig, axes = plt.subplots(rows, cols)

    # Flatten the axes if necessary
    if num_images == 1:
        axes = np.array([axes])

    # Iterate over the image list and plot each image
    for i, image in enumerate(image_list):
        ax = axes.flatten()[i]
        ax.imshow(image, cmap='gray')
        ax.axis('off')

    # Adjust the layout of subplots
    fig.tight_layout()
    return fig


def plot_intensity_images(image_list, title=''):
    latex_installed = False
    num_images = len(image_list)
    fig, axes = plt.subplots(1, num_images, figsize=(12, 2.5))

    # Flatten the axes if necessary
    if num_images == 1:
        axes = np.array([axes])

    # Iterate over the image list and plot each image
    for i, image in enumerate(image_list):
        ax = axes.flatten()[i]
        im = ax.imshow(image, cmap='gray')
        ax.axis('off')
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if latex_installed is True:
            ax.set_title(fr'$\Sigma_{i}$', usetex=True)
    plt.suptitle(f'Intensity images at various polarizer settings {title}')
    plt.rcParams.update({
        "text.usetex": latex_installed,
        "font.family": "sans-serif"
    })
    # Adjust the layout of subplots
    fig.tight_layout()
    return fig
