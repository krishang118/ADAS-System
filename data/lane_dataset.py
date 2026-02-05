import cv2
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class CULaneDataset(Dataset):
    
    def __init__(self, root_dir, split='train', img_height=288, img_width=800):
        
        self.root_dir = Path(root_dir)
        self.split = split
        self.img_height = img_height
        self.img_width = img_width
        
        list_file = self.root_dir / "list" / f"{split}_gt.txt"
        
        if not list_file.exists():
            print(f"Warning: List file {list_file} not found")
            self.samples = []
        else:
            with open(list_file, 'r') as f:
                self.samples = []
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        img_path = parts[0].lstrip('/')  
                        mask_path = parts[1].lstrip('/')  
                        self.samples.append((img_path, mask_path))
        
        print(f"Loaded {len(self.samples)} samples for {split}")
        
        if split == 'train':
            self.transform = A.Compose([
                A.Resize(img_height, img_width, interpolation=cv2.INTER_LINEAR),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], additional_targets={'mask': 'mask'},
               is_check_shapes=False)
        else:
            self.transform = A.Compose([
                A.Resize(img_height, img_width, interpolation=cv2.INTER_LINEAR),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], additional_targets={'mask': 'mask'},
               is_check_shapes=False)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path_str, mask_path_str = self.samples[idx]
        
        img_path = self.root_dir / img_path_str
        mask_path = self.root_dir / mask_path_str
        
        if not img_path.exists():
            print(f"Warning: Image not found: {img_path}")
            image = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)
        else:
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Warning: Failed to load: {img_path}")
                image = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if not mask_path.exists():
            print(f"Warning: Mask not found: {mask_path}")
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        else:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"Warning: Failed to load mask: {mask_path}")
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        mask = cv2.resize(mask, (self.img_width, self.img_height), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.uint8) * 255
        
        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']
        
        mask = (mask > 127).float().unsqueeze(0)
        
        return image, mask
