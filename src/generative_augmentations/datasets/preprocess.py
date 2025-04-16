from pathlib import Path
# from dataclasses import dataclass

from torchvision import tv_tensors
import numpy as np
import torch as  th 
import cv2 
from tqdm import tqdm
from pycocotools.mask import encode


# @dataclass
# class AnnotationObject:
#     """One instance of a instance segmentation annotation."""
#     mask: tv_tensors.Mask
#     boxes: tv_tensors.BoundingBoxes
#     labels: int
        

def preprocess_annotation(img_path: Path, anno_path: Path, mode: str, img_id: str, parent: Path = Path('')) -> None:
    """
    Loads in image and annotation. Preprocesses and stores them as 
    img --> img.png
        --> annotation.pth 
    where annotation.pth is a dictionary with 
    "boxes" (tv_tensors.BoundingBoxes)
    "masks" (tv_tensors.Masks)
    "labels" (th.Tensor) ? Not sure about these last two. 
    "image_id" (int) ? 
    """
    img = cv2.imread(img_path)
    height, width = img.shape[0], img.shape[1]
    
    # Load in each row of the annotation 
    boxes = []
    masks = []
    labels = []
    with open(anno_path, 'r') as file:
        for line in file:
            numbers = list(map(float, line.split(" ")))
            labels.append(numbers[0])
            
            # Create mask 
            mask = np.zeros((height, width))
            endpoints = np.array(numbers[1:]).reshape(-1,2)
            endpoints = np.round(endpoints @ np.diag([height, width])).astype(np.int32)
            cv2.fillPoly(mask, [endpoints], color=1)
            rle_mask = encode(np.asfortranarray(mask.astype(np.uint8)))
            masks.append(rle_mask)
            
            # Create bbox 
            lower_right = np.max(endpoints, axis=0)[[1,0]] # Need to have XY not YX
            upper_left  = np.min(endpoints, axis=0)[[1,0]]
            boxes.append(np.concatenate([upper_left, lower_right]))
    
    # Convert to correct format 
    annotation = {'boxes': boxes, #tv_tensors.BoundingBoxes(boxes, format='XYXY', canvas_size=(height, width)),
                  'img_shape': (height, width), 
                  'masks': masks, #tv_tensors.Mask(masks),
                  'labels': th.Tensor(labels)}
    
    # Check that the correct folders exists
    folder = parent / mode / img_id 
    folder.mkdir(parents=True, exist_ok=True)
    
    th.save(annotation, folder / 'anno.pth')
    cv2.imwrite(str(folder / 'img.png'), img)

    text_file = parent / mode / 'image_names.txt'
    with open(text_file, 'a') as file:
        file.write(img_id + ' ') 
        
    
def preprocess_all(img_path: Path, anno_path: Path, mode: str, parent: Path = Path(''), n_samples: int = 10000):
    file_names = np.array(list(img_path.iterdir()))
    sample_files = np.random.choice(file_names, size=n_samples, replace=False)

    for img_file in tqdm(sample_files):
        anno_file = anno_path / (img_file.stem + ".txt")
        if not anno_file.exists():
            continue
        preprocess_annotation(img_path = img_file, anno_path=anno_file, mode=mode, img_id=img_file.stem, parent=parent)
        


if __name__ == "__main__": 
    preprocess_all(img_path=Path("data/raw/datatrain/train2017"), anno_path=Path("data/raw/datamasks/coco/labels/train2017"), mode="train", parent=Path("data/coco"), n_samples=10000)
    preprocess_all(img_path=Path("data/raw/dataval/val2017"), anno_path=Path("data/raw/datamasks/coco/labels/val2017"), mode="val", parent=Path("data/coco"), n_samples=1000)
