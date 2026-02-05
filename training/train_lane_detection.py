import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from config.config import Config
from data.lane_dataset import CULaneDataset
from models.lane_segmentation import get_lane_model

def dice_loss(pred, target, smooth=1.):

    pred = torch.sigmoid(pred)
    pred = pred.contiguous()
    target = target.contiguous()
    
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = 1 - ((2. * intersection + smooth) / 
                (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth))
    
    return loss.mean()

def train_lane_detection(quick_test=False, resume=False):

    Config.ensure_dirs()
    
    print("Loading CULane dataset...")
    train_dataset = CULaneDataset(
        root_dir=Config.CULANE_ROOT,
        split='train',
        img_height=Config.LANE_IMG_HEIGHT,
        img_width=Config.LANE_IMG_WIDTH
    )
    
    val_dataset = CULaneDataset(
        root_dir=Config.CULANE_ROOT,
        split='val',
        img_height=Config.LANE_IMG_HEIGHT,
        img_width=Config.LANE_IMG_WIDTH
    )
    
    if quick_test:
        print("Quick test mode: using subset of data")
        train_dataset.samples = train_dataset.samples[:500]
        val_dataset.samples = val_dataset.samples[:100]
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, 
                              shuffle=True, num_workers=Config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, 
                            shuffle=False, num_workers=Config.NUM_WORKERS)
    
    print("Initializing model...")
    model = get_lane_model(device=Config.DEVICE)
    
    best_loss = float('inf')
    start_epoch = 0
    
    if resume:
        weight_path = Config.WEIGHTS_DIR / "lane_segmentation_best.pth"
        if weight_path.exists():
            print(f"Resuming training from {weight_path}")
            model.load_state_dict(torch.load(weight_path, map_location=Config.DEVICE))
            print("Weights loaded successfully")
        else:
            print(f"Warning: Checkpoint not found at {weight_path}, starting fresh")
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)
    
    epochs = Config.LANE_EPOCHS_QUICK if quick_test else Config.LANE_EPOCHS
    best_loss = float('inf')
    
    print(f"Training for {epochs} epochs on {Config.DEVICE}...")
    
    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0
        
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images = images.to(Config.DEVICE)
            masks = masks.to(Config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            loss = criterion(outputs, masks) + dice_loss(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(Config.DEVICE)
                masks = masks.to(Config.DEVICE)
                
                outputs = model(images)
                loss = criterion(outputs, masks) + dice_loss(outputs, masks)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            save_path = Config.WEIGHTS_DIR / "lane_segmentation_best.pth"
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")
        
        scheduler.step(val_loss)
    
    print("Training completed.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train lane detection model')
    parser.add_argument('--quick', action='store_true', help='Quick test mode with subset of data')
    parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint')
    args = parser.parse_args()
    train_lane_detection(quick_test=args.quick, resume=args.resume)
