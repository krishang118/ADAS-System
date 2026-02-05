import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from config.config import Config
from models.yolo_wrapper import YOLOWrapper

class TrafficLightDetector:
    
    def __init__(self, weight_path=None):
       
        self.yolo = YOLOWrapper(model_size='n')
        
        if weight_path is None:
            weight_path = Config.WEIGHTS_DIR / 'yolo_lights_best.pt'
        
        if Path(weight_path).exists():
            self.yolo.load_weights(str(weight_path))
            self.class_names = self.yolo.model.names
            print(f"Loaded traffic light model from {weight_path}")
        else:
            print(f"Warning: Model weights not found at {weight_path}")
            self.class_names = {0: 'red', 1: 'yellow', 2: 'green', 3: 'off'}
    
    def detect(self, frame, conf_threshold=None, imgsz=None):
      
        if conf_threshold is None:
            conf_threshold = Config.LIGHT_CONF_THRESHOLD
            
        if imgsz is None:
            imgsz = Config.YOLO_IMG_SIZE
        
        results = self.yolo.predict(frame, conf=conf_threshold, imgsz=imgsz)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                
                state = self.class_names.get(cls, 'unknown')
                
                detections.append({
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'conf': conf,
                    'state': state,
                    'class_id': cls
                })
        
        return detections
