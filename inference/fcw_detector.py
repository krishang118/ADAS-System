import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from config.config import Config
from models.yolo_wrapper import YOLOWrapper
from utils.tracking import Sort

class FCWDetector:
    
    def __init__(self, weight_path=None):
      
        self.yolo = YOLOWrapper(model_size='n')
        
        if weight_path is None:
            weight_path = Config.WEIGHTS_DIR / 'yolo_fcw_best.pt'
        
        if Path(weight_path).exists():
            self.yolo.load_weights(str(weight_path))
            print(f"Loaded FCW model from {weight_path}")
        else:
            print(f"Warning: Model weights not found at {weight_path}")
        
        self.tracker = Sort(max_age=5, min_hits=3, iou_threshold=0.3)
        self.prev_boxes = {}
        self.fps = 30  
    
    def detect(self, frame, conf=None, conf_threshold=None):
       
        threshold = conf if conf is not None else conf_threshold
        if threshold is None:
            threshold = Config.FCW_CONF_THRESHOLD
        
        results = self.yolo.predict(frame, conf=threshold, imgsz=Config.YOLO_IMG_SIZE)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                
                detections.append([x1, y1, x2, y2, conf])
        
        return np.array(detections) if detections else np.empty((0, 5))
    
    def track(self, detections):
      
        if len(detections) == 0:
            return []
        
        tracked = self.tracker.update(detections)
        return tracked
    
    def calculate_ttc(self, tracked_boxes, frame_height):
      
        ttc_warnings = []
        
        for tracked in tracked_boxes:
            x1, y1, x2, y2, track_id = tracked
            
            box_area = (x2 - x1) * (y2 - y1)
            box_height = y2 - y1
            
            if track_id in self.prev_boxes:
                prev_area = self.prev_boxes[track_id]['area']
                prev_height = self.prev_boxes[track_id]['height']
                
                area_growth = box_area - prev_area
                height_growth = box_height - prev_height
                
                if height_growth > 0:
                    ttc = (box_height / height_growth) * (1.0 / self.fps)
                    ttc = max(0, ttc) 
                else:
                    ttc = float('inf')
                
                if ttc < Config.FCW_TTC_THRESHOLD and ttc > 0:
                    ttc_warnings.append({
                        'track_id': track_id,
                        'bbox': [x1, y1, x2, y2],
                        'ttc': ttc,
                        'warning': True
                    })
                else:
                    ttc_warnings.append({
                        'track_id': track_id,
                        'bbox': [x1, y1, x2, y2],
                        'ttc': ttc,
                        'warning': False
                    })
            
            self.prev_boxes[track_id] = {
                'area': box_area,
                'height': box_height
            }
        
        return ttc_warnings
    
    def set_fps(self, fps):
        self.fps = fps
