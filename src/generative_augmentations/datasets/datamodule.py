from pathlib import Path 

import lightning as pl
import torch as th 
from torch.utils.data import Dataset, DataLoader
import cv2 
from pycocotools.mask import decode
from torchvision import tv_tensors

class COCODataset(Dataset): 
    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path 
        with open(path / 'image_names.txt', 'r') as file:
            line = file.readline().strip()
        self.folders = line.split(' ')
    
    def __len__(self): 
        return len(self.folders)
    
    def __getitem__(self, i): 
        folder = self.folders[i]
        img = cv2.imread(self.path / (folder + "/img.png"))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        annotations = th.load(self.path / (folder + "/anno.pth"))
        
        masks = annotations["masks"]
        masks_full = [th.tensor(decode(mask)) for mask in masks]
        annotations['masks'] = tv_tensors.Mask(masks_full)
        
        return img, annotations
    
    
class COCODataModule(pl.LightningDataModule):
    def __init__(self, data_dir: Path = Path('./data/coco'), batch_size: int = 32, num_workers: int = 16, transforms=None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.setup()

    def setup(self, stage=None):
        self.train_set = COCODataset(self.data_dir / 'train')
        # self.val_set = COCODataset(self.data_dir / 'val')

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    # def val_dataloader(self):
    #     return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers)

    # def test_dataloader(self):
    #     return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers)

from torch.utils.data import random_split
from torchvision.datasets import MNIST

class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "path/to/dir", batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage: str):
        self.mnist_test = MNIST(self.data_dir, train=False)
        self.mnist_predict = MNIST(self.data_dir, train=False)
        mnist_full = MNIST(self.data_dir, train=True)
        self.mnist_train, self.mnist_val = random_split(
            mnist_full, [55000, 5000], generator=th.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=self.batch_size)