from pathlib import Path 

import pytorch_lightning as pl
import torch as th 
from torch.utils.data import Dataset, DataLoader
import cv2 

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
        img = cv2.imread(folder + "/img.png")
        annotations = th.load(folder + "/anno.pth")
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
        self.val_set = COCODataset(self.data_dir / 'val')

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers)

    # def test_dataloader(self):
    #     return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers)
