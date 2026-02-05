import torch
import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from config.config import Config
from models.lane_segmentation import get_lane_model

class LaneDetector:
    
    def __init__(self, weight_path=None, device=None):
    
        self.device = device or Config.DEVICE
        self.model = get_lane_model(device=self.device)
        
        if weight_path is None:
            weight_path = Config.WEIGHTS_DIR / 'lane_segmentation_best.pth'
        
        if Path(weight_path).exists():
            self.model.load_state_dict(torch.load(weight_path, map_location=self.device))
            self.model.eval()
            print(f"Loaded lane detection model from {weight_path}")
        else:
            print(f"Warning: Model weights not found at {weight_path}")
    
    def preprocess(self, frame):
        img = cv2.resize(frame, (Config.LANE_IMG_WIDTH, Config.LANE_IMG_HEIGHT))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        
        img = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)
        return img.to(self.device)
    
    def detect(self, frame):
    
        with torch.no_grad():
            img_tensor = self.preprocess(frame)
            output = self.model(img_tensor)
            mask = torch.sigmoid(output).cpu().numpy()[0, 0]
        
        mask = (mask > 0.5).astype(np.uint8) * 255
        
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
        
        return mask
    
    def fit_lanes(self, mask):
      
        lane_pixels = np.where(mask > 127)
        
        if len(lane_pixels[0]) < 10:
            return None, None
        
        midpoint = mask.shape[1] // 2
        
        left_mask = lane_pixels[1] < midpoint
        right_mask = lane_pixels[1] >= midpoint
        
        left_lane = (lane_pixels[0][left_mask],
                     lane_pixels[1][left_mask])
        
        right_lane = (lane_pixels[0][right_mask],
                      lane_pixels[1][right_mask])
        
        left_fit = None
        right_fit = None
        
        if len(left_lane[0]) > 10:
            left_fit = np.polyfit(left_lane[0].astype(np.float32), left_lane[1].astype(np.float32), 2)
        
        if len(right_lane[0]) > 10:
            right_fit = np.polyfit(right_lane[0].astype(np.float32), right_lane[1].astype(np.float32), 2)
        
        return left_fit, right_fit
    
    def calculate_offset(self, left_fit, right_fit, frame_height, frame_width):
      
        if left_fit is None or right_fit is None:
            return None
        
        left_fit = np.array(left_fit, dtype=np.float32)
        right_fit = np.array(right_fit, dtype=np.float32)
        
        y = np.float32(frame_height - 1)
        left_x = left_fit[0]*y**2 + left_fit[1]*y + left_fit[2]
        right_x = right_fit[0]*y**2 + right_fit[1]*y + right_fit[2]
        
        lane_center = (left_x + right_x) / 2
        
        vehicle_center = np.float32(frame_width) / 2
        
        offset_pixels = vehicle_center - lane_center
        
        lane_width = right_x - left_x
        if lane_width > 0:
            offset_ratio = offset_pixels / lane_width
        else:
            offset_ratio = np.float32(0)
        
        return float(offset_ratio)
    
    def check_departure_warning(self, offset_ratio, threshold=None):
      
        if offset_ratio is None:
            return False, "unknown"
        
        thresh = threshold if threshold is not None else Config.LDW_OFFSET_THRESHOLD
        
        if abs(offset_ratio) > thresh:
            direction = "left" if offset_ratio < 0 else "right"
            return True, direction
        
        return False, "none"
