from typing import Literal
from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2


type TransformTypes = Literal["final transform", "simple augmentation", "advanced augmentation"]
transforms: dict[TransformTypes, A.Compose] = {}


cv2_border_replicate = 1


transforms["final transform"] = A.Compose([
    A.Resize(224, 224),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),
    ToTensorV2()
])


transforms["simple augmentation"] = A.Compose([
    A.HorizontalFlip(),
    # NOTE: Prefering to rotate then crop, because this is more likely to leave center crops with no black borders.
    A.Rotate(limit=30, border_mode=cv2_border_replicate),
    A.RandomCropFromBorders(crop_bottom=0.2, crop_left=0.2, crop_right=0.2, crop_top=0.2),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms["final transform"],
])


transforms["advanced augmentation"] = A.Compose([ # pyright: ignore[reportArgumentType]
    A.MotionBlur(blur_limit=(3, 5)),
    A.OpticalDistortion(distort_limit=0.4, border_mode=cv2_border_replicate),
    A.ThinPlateSpline(scale_range=(0.03, 0.06), border_mode=cv2_border_replicate),
    A.HorizontalFlip(),
    A.Affine(scale=(0.9, 1.2), translate_percent=(0.0, 0.1), shear=(-7, 7), rotate=(-5, 5), rotate_method="ellipse", border_mode=cv2_border_replicate),
    # NOTE: Prefering to rotate then crop, because this is more likely to leave center crops with no black borders.
    A.RandomCropFromBorders(crop_bottom=0.2, crop_left=0.2, crop_right=0.2, crop_top=0.2),

    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    A.RandomGamma(gamma_limit=(80, 140)),
    A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.1, 0.3)),
    A.Downscale(scale_range=(0.6, 0.9)),
    A.ImageCompression(quality_range=(40, 80)),
    
    transforms["final transform"]
])




if __name__ == "__main__":
    # Only import when running this file as a script
    import tyro
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.colors import Normalize
    from matplotlib.patches import Patch
    import numpy as np
    from lightning.pytorch import seed_everything
    
    from src.generative_augmentations.config import Config
    from src.generative_augmentations.datasets.datamodule import COCODataModule
    from src.generative_augmentations.datasets.coco import index_to_name

    # Set up settings
    config = tyro.cli(Config)

    seed_everything(config.seed)
    
    # Next image on spacebar
    plt.rcParams['keymap.quit'].append(' ')



    # Load datamodule
    datamodule = COCODataModule(
        num_workers=1,
        batch_size=1,
        transform_train=transforms["advanced augmentation"],
        transform_val=transforms["final transform"],
        data_fraction=config.dataloader.data_fraction,
        data_dir=Path(config.dataloader.data_dir)
    )

    datamodule.setup()
    datamodule.train_set.dataset.include_original_image = True # pyright: ignore[ reportAttributeAccessIssue ]

    # Set up standardization values
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # Set up colors
    cmap = mpl.colormaps["nipy_spectral"]
    norm = Normalize(vmin=min(index_to_name), vmax=max(index_to_name))


    # Get a batch from the training dataloader
    for img, annotations in datamodule.train_dataloader():
        # Extract items
        img = img[0].cpu().numpy().transpose(1, 2, 0)
        img = np.clip(std * img + mean, 0, 1)

        img_original = annotations[0]["image"]

        mask = annotations[0]["semantic_mask"].cpu().numpy()

        name = annotations[0]["name"]
        
        # Create legend labels
        label_ids = np.unique(mask).tolist()
        label_ids.sort()
        label_names = [index_to_name[idx] for idx in label_ids]
        legend = [
            Patch(facecolor=cmap(norm(idx)), edgecolor="black", label=f"{name} ({idx})") 
            for idx, name in zip(label_ids, label_names)
        ]

        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 6))
        
        axes[0].imshow(img_original)
        axes[0].set_title("Image - Original")
        axes[0].axis("off")

        axes[1].imshow(img)
        axes[1].set_title("Image - Transformed")
        axes[1].axis("off")

        axes[2].imshow(mask, cmap=cmap, norm=norm)
        axes[2].set_title("Semantic Mask")
        axes[2].axis("off")

        fig.legend(
            handles=legend, 
            loc='upper center', 
            bbox_to_anchor=(0.5, 1.02), 
            ncol=min(len(legend), 5), 
            title=name
        )

        plt.tight_layout(rect=(0, 0, 1, 0.9))
        plt.show()
