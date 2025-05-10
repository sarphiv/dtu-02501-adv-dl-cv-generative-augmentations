from pathlib import Path 
from typing import Callable
from collections import defaultdict

import lightning as pl
from tqdm import tqdm 
import numpy as np
import torch as th 
from torch.utils.data import Dataset, DataLoader, Subset
import cv2 
from pycocotools.mask import decode
from torchvision import tv_tensors
from torchvision.transforms.functional import resize
from albumentations import BaseCompose

from src.generative_augmentations.datasets.coco import index_to_name


def collate_img_anno(batch): 
    images = th.stack([item[0] for item in batch])
    annotations = [item[1] for item in batch]
    return images, annotations


class COCODatasetv2(Dataset): 
    def __init__(self, path: Path, transform: BaseCompose | None = None, augmentation_diffusion: tuple[float, int] | None = None, augmentation_instance_prob: float | None = None, include_original_image: bool = False, pin_images_mem: bool = False) -> None: 
        super().__init__()
        self.path = path 
        self.transform = transform
        self.augmentation_diffusion = augmentation_diffusion
        self.augmentation_instance_prob = augmentation_instance_prob
        self.include_original_image = include_original_image
        self.pin_images_mem = pin_images_mem
        with open(path / 'image_names.txt', 'r') as file:
            line = file.readline().strip()
        self.folders = line.split(' ')
        
        if pin_images_mem: 
            self.images = [self._get_img_from_file(self.path / folder / "img.png") for folder in tqdm(self.folders)] 
                
    def _get_img_from_file(self, file_path: Path):
        img = cv2.imread(str(file_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
        
    def __len__(self): 
        return len(self.folders)
    
    def __getitem__(self, i): 
        # Loading image and annotation 
        folder = self.folders[i]
        img = self.images[i] if self.pin_images_mem else self._get_img_from_file(self.path / folder / "img.png")
        annotations = th.load(self.path / folder / "anno.pth", weights_only=False)
        
        masks = annotations["masks"]
        masks_full = [decode(mask) for mask in masks]
        if self.include_original_image: 
            annotations['image'] = img
        boxes = annotations['boxes']
        labels = annotations['labels'].numpy()

        if self.augmentation_instance_prob is not None: 
            # Load which instances have augmentations 
            instances = defaultdict(list)
            with open(self.path / folder / 'variants.txt', 'r') as f:
                for line in f: 
                    entries = line.strip().split()
                    instances[entries[0]].append(entries[1])
            # For each instance choose an augmentation or to keep the original 
            instance_choices = np.zeros(len(labels), dtype=int) - 1
            choices = np.random.binomial(1, size=len(instance_choices), p=self.augmentation_instance_prob)
            augmentations = []
            for i in range(len(instance_choices)): 
                if choices[i] and str(i) in instances.keys(): 
                    aug = np.random.choice(instances[str(i)])
                    aug = cv2.imread(self.path / folder / 'variants' / f'{i}_{aug}.png')
                    aug = cv2.cvtColor(aug, cv2.COLOR_BGR2RGB)
                    (x_1, y_1, x_2, y_2) = boxes[i]
                    if aug.shape != (y_2-y_1, x_2-x_1, 3): 
                        aug = cv2.resize(aug, dsize=(x_2-x_1, y_2-y_1,))
                    padded_aug = np.zeros_like(img)
                    padded_aug[y_1:y_2, x_1:x_2] = aug

                    instance_choices[i] = len(augmentations)
                    augmentations.append(padded_aug)
        
        # Generate image and collect masks for each class 
        area = th.tensor([int(mask.sum()) for mask in masks_full]).argsort(descending=True)
        segmentation_mask = np.zeros((img.shape[0], img.shape[1]), dtype=int) + len(index_to_name) - 1
        new_image = img.copy() 

        if self.augmentation_diffusion and np.random.uniform() <= self.augmentation_diffusion[0]: 
            variant_idx = np.random.randint(self.augmentation_diffusion[1])
            new_image = self._get_img_from_file(self.path / folder / f'img2img' / f'{variant_idx}.png')

        for a in area:
            segmentation_mask[masks_full[a] != 0] = int(labels[a]) 
            if self.augmentation_instance_prob is not None: 
                instance = instance_choices[a]
                instance = img if instance == -1 else augmentations[instance]
                new_image[masks_full[a] != 0] = instance[masks_full[a] != 0]


        if self.transform:
            augmented = self.transform(
                image=new_image,
                masks=[segmentation_mask]
            )
            new_image = augmented['image']
            segmentation_mask = augmented['masks'][0]

        annotations['img_shape'] = (new_image.shape[1], new_image.shape[2])
        annotations['semantic_mask'] = segmentation_mask
        annotations['boxes'] = tv_tensors.BoundingBoxes(np.array(boxes), format='XYXY', canvas_size=annotations['img_shape'])
        annotations['labels'] = th.tensor(labels.astype(int))
        annotations['name'] = folder
    
        return new_image, annotations
    
class COCODataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_dir: Path = Path('./data/coco'),
            batch_size: int = 32,
            num_workers: int = 16,
            transform_train: BaseCompose | None = None,
            transform_val: BaseCompose | None = None,
            augmentation_instance_prob: float | None = None,
            augmentation_diffusion: tuple[float, int] | None = None,
            data_fraction: float = 1,
            pin_images_mem: bool = False
        ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.transform_train = transform_train
        self.transform_val = transform_val
        self.augmentation_instance_prob = augmentation_instance_prob
        self.augmentation_diffusion = augmentation_diffusion

        self.data_fraction = data_fraction
        self.pin_images_mem = pin_images_mem
        
        self.setup2()

    def setup2(self, stage=None):
        train_set = COCODatasetv2(
            self.data_dir / 'train', 
            transform=self.transform_train, 
            augmentation_diffusion=self.augmentation_diffusion, 
            augmentation_instance_prob=self.augmentation_instance_prob, 
            pin_images_mem=self.pin_images_mem
        )
        n_train = len(train_set)
        train_idx = th.randperm(n_train)[:int(n_train*self.data_fraction)].tolist()
        
        self.train_set = Subset(train_set, train_idx)
        self.val_set = COCODatasetv2(
            self.data_dir / 'val', 
            transform=self.transform_val, 
            include_original_image=True,
            pin_images_mem=self.pin_images_mem
        )

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=collate_img_anno, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=collate_img_anno)

    # def test_dataloader(self):
    #     return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers)


if __name__ == "__main__": 
    from tqdm import tqdm

    dm = COCODataModule(data_dir=Path('C:/Users/david/Desktop/coco/train'), batch_size=2, num_workers=0, augmentation_instance_prob=0.2)
    dm.setup()
    dl = dm.train_dataloader()

    unique_labels = set()
    for batch in tqdm(dl):
        images = batch[0]
        annotations = batch[1]
        for annotation in annotations:
            new_labels = set(annotation['labels'].detach().numpy())
            unique_labels = unique_labels.union(new_labels)

    len(unique_labels) # This gives us 80