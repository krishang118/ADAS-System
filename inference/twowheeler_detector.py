import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from config.config import Config
from models.yolo_wrapper import YOLOWrapper

class TwoWheelerDetector:
    
    BICYCLE_CLASS = 1
    MOTORCYCLE_CLASS = 3
    
    def __init__(self):
       
        self.yolo = YOLOWrapper(model_size='n')
    
    def detect(self, frame, conf=None, conf_threshold=None):
      
        threshold = conf if conf is not None else conf_threshold
        if threshold is None:
            threshold = Config.TWOWHEELER_CONF_THRESHOLD
        
        results = self.yolo.predict(frame, conf=threshold, imgsz=Config.YOLO_IMG_SIZE)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0].cpu().numpy())
                if cls in [self.BICYCLE_CLASS, self.MOTORCYCLE_CLASS]:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    
                    vehicle_type = "Bicycle" if cls == self.BICYCLE_CLASS else "Motorcycle"
                    
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'conf': conf,
                        'type': vehicle_type,
                        'class_id': cls
                    })
        
        return detections
    
    def check_warning(self, detections, frame_height, frame_width):
        
        warnings = []
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            y_center = (y1 + y2) / 2
            
            if bbox_height > Config.TWOWHEELER_DISTANCE_THRESHOLD or y_center > frame_height * 0.4:
                warnings.append({
                    'bbox': det['bbox'],
                    'conf': det['conf'],
                    'type': det['type'],
                    'warning': True
                })
            else:
                warnings.append({
                    'bbox': det['bbox'],
                    'conf': det['conf'],
                    'type': det['type'],
                    'warning': False
                })
        
        return warnings
