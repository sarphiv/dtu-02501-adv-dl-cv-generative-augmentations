import matplotlib.pyplot as plt 

import numpy as np

def plot_segmentation(image, target, detection):
    fig, ax = plt.subplots(2, 2, figsize=(20, 20))
    image = image.detach().cpu().permute(1,2,0).numpy()
    image = (image - image.min())/(image.max() - image.min())
    ax[0,0].imshow(image)
    ax[0,0].set_title("(Unnormalized) Augmented Image")

    original_image = target['image'].detach().cpu().numpy()
    ax[0,1].imshow(original_image)
    ax[0,1].set_title("Original Image")
 
    masks_target = target["masks"].detach().cpu().numpy()
    masks_detection = detection["masks"].detach().cpu().numpy() > 0.5

    N = max(len(masks_target), len(masks_detection))
    color_mask_target = np.zeros_like(image)
    color_mask_detection = np.zeros_like(image)
    if N <= 20: 
        cmap = plt.get_cmap('tab20', N)  # or 'viridis', 'plasma', etc.
        colors = np.array(cmap(range(N)))  # RGB, ignore alpha
    else : 
        cmap = plt.get_cmap('tab20', 20)
        colors = [cmap(i % 20) for i in range(N)]

    for i, mask in enumerate(masks_detection): 
        color_mask_detection[mask.astype(bool)[0]] += np.array(colors[i])[:-1]
    color_mask_detection /= masks_detection.sum(axis=0)[0,:,:, None] + 1e-10

    for i, mask in enumerate(masks_target):
        color_mask_target[mask.astype(bool)] += np.array(colors[i])[:-1]
    color_mask_target /= masks_target.sum(axis=0)[:,:, None] + 1e-10
        
    ax[1, 0].imshow(0.3*image+0.7*color_mask_target)
    ax[1, 0].set_title("Target Segmentation")
    ax[1, 1].imshow(0.3*image+0.7*color_mask_detection)
    ax[1, 1].set_title("Predicted Segmentation")

    return fig 

    