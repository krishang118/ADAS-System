from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from config.config import Config
from models.drowsiness_detector import DrowsinessDetector

class DrowsinessMonitor:
    
    def __init__(self, weight_path=None):
       
        self.detector = DrowsinessDetector(
            ear_threshold=Config.EAR_THRESHOLD_ALERT,
            drowsy_threshold=Config.EAR_THRESHOLD_DROWSY,
            blink_frames=None, 
            drowsy_frames=None, 
            window_size=int(Config.WINDOW_TIME_SEC * 30) 
        )
        print("Drowsiness monitor initialized (MediaPipe Tasks API)")
    
    def detect(self, frame, ear_alert=None, ear_drowsy=None, drowsy_frames=None, drowsy_time_sec=None, fps=30.0, timestamp_ms=None):
    
        if ear_alert is not None:
            self.detector.ear_threshold = ear_alert
        if ear_drowsy is not None:
            self.detector.drowsy_threshold = ear_drowsy
        if drowsy_time_sec is not None:
            self.detector.drowsy_time_threshold_sec = drowsy_time_sec
        
        return self.detector.process_frame(frame, timestamp_ms=timestamp_ms)
    
    def reset(self):
        self.detector.reset()
    
    def draw_visualization(self, frame, result):
      
        import cv2
        from utils.visualization import Visualizer

        if result.get('face_detected'):
            if 'landmarks' in result and result['landmarks']:
                for (x, y) in result['landmarks']:
                    cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)

            ear = result.get('ear')
            state = result.get('state', 'Awake')
            blink_count = result.get('blink_count', 0)
            
            info_text = [
                f"State: {state}",
                f"EAR: {ear:.3f}" if ear is not None else "EAR: N/A",
                f"Blinks: {blink_count}"
            ]
            
            if result.get('warning'):
                frame = Visualizer.draw_warning(frame, "DRIVER DROWSINESS DETECTED")
        
        return frame
