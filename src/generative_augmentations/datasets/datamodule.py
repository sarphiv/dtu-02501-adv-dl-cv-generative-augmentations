from pathlib import Path 
from typing import Callable

import lightning as pl
import torch as th 
from torch.utils.data import Dataset, DataLoader, Subset
import cv2 
from pycocotools.mask import decode
from torchvision import tv_tensors

def collate_img_anno(batch): 
    images = [item[0] for item in batch]
    annotations = [item[1] for item in batch]
    return images, annotations

class COCODataset(Dataset): 
    def __init__(self, path: Path, transform=None, include_original_image=False) -> None: 
        super().__init__()
        self.path = path 
        self.transform = transform
        self.include_original_image = include_original_image
        with open(path / 'image_names.txt', 'r') as file:
            line = file.readline().strip()
        self.folders = line.split(' ')
    
    def __len__(self): 
        return len(self.folders)
    
    def __getitem__(self, i): 
        folder = self.folders[i]
        img = cv2.imread(self.path / (folder + "/img.png"))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        annotations = th.load(self.path / (folder + "/anno.pth"), weights_only=False)
        
        masks = annotations["masks"]
        masks_full = [decode(mask) for mask in masks]
        boxes = annotations['boxes']
        class_labels = annotations['labels'].numpy()

        if self.include_original_image: annotations['image'] = img
        if self.transform:
            augmented = self.transform(
                image=img,
                masks=masks_full,
                bboxes=boxes,
                class_labels=annotations['labels'].numpy()
            )
            img = augmented['image']
            masks_full = th.stack(augmented['masks'])
            boxes = augmented['bboxes']
            class_labels = augmented['class_labels']

        annotations['img_shape'] = (img.shape[1], img.shape[2])
        annotations['masks'] = tv_tensors.Mask(data=masks_full)
        annotations['boxes'] = tv_tensors.BoundingBoxes(boxes, format='XYXY', canvas_size=annotations['img_shape'])
        annotations['labels'] = th.tensor(class_labels.astype(int))
    
        return img, annotations
    
class COCODataModule(pl.LightningDataModule):
    def __init__(self, data_dir: Path = Path('./data/coco'), batch_size: int = 32, num_workers: int = 16, transform=None, data_fraction: float = 1):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.data_fraction = data_fraction
        
        self.setup()

    def setup(self, stage=None):
        train_set = COCODataset(self.data_dir / 'train', transform=self.transform)
        n_train = len(train_set)
        train_idx = th.randperm(n_train)[:int(n_train*self.data_fraction)].tolist()
        
        self.train_set = Subset(train_set, train_idx)
        self.val_set = COCODataset(self.data_dir / 'val', transform=self.transform, include_original_image=True)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=collate_img_anno)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=collate_img_anno)

    # def test_dataloader(self):
    #     return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers)


if __name__ == "__main__": 
    from tqdm import tqdm

    dm = COCODataModule(batch_size=2, num_workers=0)
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