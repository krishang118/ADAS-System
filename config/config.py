import os
from pathlib import Path

class Config:
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_ROOT = Path("/Volumes/drive/ADAS Project/Datasets")
    WEIGHTS_DIR = PROJECT_ROOT / "weights"    
    CULANE_ROOT = DATA_ROOT / "CULane"
    MTSD_ROOT = DATA_ROOT / "MTSD"
    LISA_ROOT = DATA_ROOT / "LISA Traffic Light Dataset"
    YOLO_FCW_DATA = DATA_ROOT / "yolo_fcw"
    YOLO_PED_DATA = DATA_ROOT / "yolo_pedestrian"
    YOLO_SIGNS_DATA = DATA_ROOT / "yolo_signs"
    YOLO_LIGHTS_DATA = DATA_ROOT / "yolo_lights"
    
    LANE_IMG_HEIGHT = 288
    LANE_IMG_WIDTH = 800
    YOLO_IMG_SIZE = 640
    
    BATCH_SIZE = 16
    NUM_WORKERS = 4
    LEARNING_RATE = 0.001
    DEVICE = "mps"  
    
    LANE_EPOCHS = 50
    LANE_EPOCHS_QUICK = 5
    
    YOLO_EPOCHS = 100
    YOLO_EPOCHS_QUICK = 10
    
    FCW_TTC_THRESHOLD = 2.5
    FCW_CONF_THRESHOLD = 0.5
    PED_CONF_THRESHOLD = 0.6
    PED_DISTANCE_THRESHOLD = 100  
    SIGN_CONF_THRESHOLD = 0.20   
    LIGHT_CONF_THRESHOLD = 0.18   
    LDW_OFFSET_THRESHOLD = 0.3    
    TWOWHEELER_CONF_THRESHOLD = 0.32  
    TWOWHEELER_DISTANCE_THRESHOLD = 80  
    ANIMAL_CONF_THRESHOLD = 0.35 
    ANIMAL_DISTANCE_THRESHOLD = 60     
    UTA_RLDD_ROOT = DATA_ROOT / "UTA-RLDD"
    
    FACE_DETECTION_CONFIDENCE = 0.5
    LANDMARK_MODEL = "mediapipe"
    
    EAR_THRESHOLD_ALERT = 0.25         
    EAR_THRESHOLD_DROWSY = 0.21        
    
    BLINK_TIME_THRESHOLD = 0.15        
    DROWSY_TIME_THRESHOLD = 1.5        
    FATIGUE_TIME_THRESHOLD = 1.0       
    
    WINDOW_TIME_SEC = 10.0             
    BLINK_RATE_ALERT_THRESHOLD = 12    
    BLINK_RATE_DROWSY_THRESHOLD = 3    
    
    DSM_TARGET_FPS = 10
    DSM_FRAME_WIDTH = 640             
    DSM_FRAME_HEIGHT = 480            
    
    DRIVER_STATES = ['Alert', 'Fatigued', 'Drowsy']
     
    PHONE_CAMERA_URL = "http://192...:4747/video"
    FRAME_SKIP = 2
    
    @classmethod
    def ensure_dirs(cls):
        cls.WEIGHTS_DIR.mkdir(exist_ok=True, parents=True)
        for data_dir in [cls.YOLO_FCW_DATA, cls.YOLO_PED_DATA, 
                         cls.YOLO_SIGNS_DATA, cls.YOLO_LIGHTS_DATA]:
            data_dir.mkdir(exist_ok=True, parents=True)
