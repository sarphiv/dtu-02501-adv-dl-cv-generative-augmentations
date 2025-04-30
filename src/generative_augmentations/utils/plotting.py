import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches

import numpy as np

from src.generative_augmentations.datasets.coco import index_to_name

def plot_segmentation(image, target, detection, class_names=index_to_name):
    fig, ax = plt.subplots(1,3, figsize=(21, 7))
    fig.tight_layout()

    # Get masks and image
    image = image.detach().cpu().permute(1,2,0).numpy()
    image = (image - image.min())/(image.max() - image.min())

    ax[0].imshow(target['image'])
    ax[0].set_title("Original Image")
    ax[0].axis('off')

    masks_target = target['semantic_mask'].detach().cpu().numpy()
    masks_detection = detection.detach().cpu().numpy()
    mask_predictions = masks_detection.argmax(axis=0) 

    # Get the unique labels 
    labels_in_play = np.unique(np.stack((mask_predictions, masks_target))) 

    N = len(labels_in_play)
    color_mask_target = np.zeros_like(image)
    color_mask_detection = np.zeros_like(image)
    if N <= 20: 
        cmap = plt.get_cmap('tab20', N)  # or 'viridis', 'plasma', etc.
        colors = np.array(cmap(range(N)))  # RGB, ignore alpha
    else : 
        cmap = plt.get_cmap('tab20', 20)
        colors = [cmap(i % 20) for i in range(N)]

    for i, label in enumerate(labels_in_play): 
        color_mask_detection[np.where(mask_predictions == label)] = np.array(colors[i])[:-1]
        color_mask_target[np.where(masks_target == label)] = np.array(colors[i])[:-1]
        
    # color_mask_detection /= masks_detection.sum(axis=0)[0,:,:, None] + 1e-10

    # for i, mask in enumerate(masks_target):
    #     color_mask_target[mask.astype(bool)] += np.array(colors[i])[:-1]
    # color_mask_target /= masks_target.sum(axis=0)[:,:, None] + 1e-10
        
    ax[1].imshow(0.3*image+0.7*color_mask_target)
    ax[1].set_title("Target Segmentation")
    ax[1].axis('off')
    ax[2].imshow(0.3*image+0.7*color_mask_detection)
    ax[2].set_title("Predicted Segmentation")
    ax[2].axis('off')

    legend_handles = [
        mpatches.Patch(color=colors[i], label=class_names[label])
        for i, label in enumerate(labels_in_play)
    ]

    fig.legend(handles=legend_handles, loc='upper center', ncol=len(legend_handles))

    return fig 

 
# def plot_segmentation(image, target, detection, class_names=index_to_name):
#     fig, ax = plt.subplots(2,2, figsize=(20, 20))
#     image = image.detach().cpu().permute(1,2,0).numpy()
#     image = (image - image.min())/(image.max() - image.min())
#     ax[0,0].imshow(image)
#     ax[0,0].set_title("(Unnormalized) Augmented Image")

#     ax[0,1].imshow(original_image)
#     ax[0,1].set_title("Original Image")
 
#     masks_target = target['semantic_mask'].detach().cpu().numpy()
#     masks_detection = detection.detach().cpu().numpy()
#     mask_predictions = masks_detection.argmax(axis=0) 

#     # Get the unique labels 
#     labels_in_play = np.unique(np.stack((mask_predictions, masks_target))) 

#     N = len(labels_in_play)
#     color_mask_target = np.zeros_like(image)
#     color_mask_detection = np.zeros_like(image)
#     if N <= 20: 
#         cmap = plt.get_cmap('tab20', N)  # or 'viridis', 'plasma', etc.
#         colors = np.array(cmap(range(N)))  # RGB, ignore alpha
#     else : 
#         cmap = plt.get_cmap('tab20', 20)
#         colors = [cmap(i % 20) for i in range(N)]
    
#     for i, label in enumerate(labels_in_play): 
#         color_mask_detection[np.where(mask_predictions == label)] = np.array(colors[i])[:-1]
#         color_mask_target[np.where(masks_target == label)] = np.array(colors[i])[:-1]
        
#     # color_mask_detection /= masks_detection.sum(axis=0)[0,:,:, None] + 1e-10

#     # for i, mask in enumerate(masks_target):
#     #     color_mask_target[mask.astype(bool)] += np.array(colors[i])[:-1]
#     # color_mask_target /= masks_target.sum(axis=0)[:,:, None] + 1e-10
        
#     ax[1, 0].imshow(0.3*image+0.7*color_mask_target)
#     ax[1, 0].set_title("Target Segmentation")
#     ax[1, 1].imshow(0.3*image+0.7*color_mask_detection)
#     ax[1, 1].set_title("Predicted Segmentation")

#     legend_handles = [
#         mpatches.Patch(color=colors[i], label=class_names[label])
#         for i, label in enumerate(labels_in_play)
#     ]

#     fig.legend(handles=legend_handles, loc='upper center', ncol=len(legend_handles))

#     return fig 

    