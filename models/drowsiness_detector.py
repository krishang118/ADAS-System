import cv2
import numpy as np
from collections import deque
from scipy.spatial import distance
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path
import sys
import os

sys.path.append(str(Path(__file__).parent.parent))

from config.config import Config

class DrowsinessDetector:
  
    def __init__(self, ear_threshold=None, drowsy_threshold=None,
                 blink_frames=None, drowsy_frames=None, window_size=None, drowsy_time_threshold=None):
        
        self.ear_threshold = ear_threshold or Config.EAR_THRESHOLD_ALERT
        self.drowsy_threshold = drowsy_threshold or Config.EAR_THRESHOLD_DROWSY
        self.blink_frames_threshold = blink_frames or int(Config.BLINK_TIME_THRESHOLD * 30)
        self.drowsy_frames_threshold = drowsy_frames or int(Config.DROWSY_TIME_THRESHOLD * 30)
        self.drowsy_time_threshold_sec = drowsy_time_threshold or Config.DROWSY_TIME_THRESHOLD
        self.window_size = window_size or int(Config.WINDOW_TIME_SEC * 30)        
        base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            min_face_detection_confidence=Config.FACE_DETECTION_CONFIDENCE,
            min_face_presence_confidence=Config.FACE_DETECTION_CONFIDENCE,
            min_tracking_confidence=Config.FACE_DETECTION_CONFIDENCE
        )
        
        model_path = Path('face_landmarker.task')
        if not model_path.exists():
            print("Face Landmarker model not found. Downloading...")
            self._download_model()

        self.landmarker = vision.FaceLandmarker.create_from_options(options)        
        self.ear_history = deque(maxlen=self.window_size)
        self.blink_history = deque(maxlen=self.window_size)
        self.current_state = 'Alert'
        self.state_counter = 0
        self.frame_counter = 0  
        self.blink_counter = 0       
        self.last_timestamp_ms = None
        self.last_timestamp_ms = None
        self.closed_start_time = None 
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        
    def _download_model(self):
        import urllib.request
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        try:
            urllib.request.urlretrieve(url, "face_landmarker.task")
            print("Model downloaded successfully")
        except Exception as e:
            print(f"Failed to download model: {e}")
            raise RuntimeError("Could not download face_landmarker.task")

    def calculate_ear(self, eye_landmarks):
        A = distance.euclidean(eye_landmarks[1], eye_landmarks[5])
        B = distance.euclidean(eye_landmarks[2], eye_landmarks[4])
        C = distance.euclidean(eye_landmarks[0], eye_landmarks[3])
        ear = (A + B) / (2.0 * C + 1e-6)
        return ear
    
    def extract_eye_landmarks(self, face_landmarks, image_shape):
        h, w = image_shape[:2]
        
        left_eye = []
        for idx in self.LEFT_EYE:
            landmark = face_landmarks[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            left_eye.append((x, y))
        
        right_eye = []
        for idx in self.RIGHT_EYE:
            landmark = face_landmarks[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            right_eye.append((x, y))
        
        return np.array(left_eye), np.array(right_eye)
    
    def update_state(self, ear, timestamp_ms):
      
        if self.last_timestamp_ms is None:
            self.last_timestamp_ms = timestamp_ms
            self.closed_start_time = None
            return 'Alert'
            
        dt = timestamp_ms - self.last_timestamp_ms
        self.last_timestamp_ms = timestamp_ms
        
        if ear < self.drowsy_threshold:
            if self.closed_start_time is None:
                self.closed_start_time = timestamp_ms
            
            closed_duration = timestamp_ms - self.closed_start_time
            
            drowsy_threshold_ms = self.drowsy_time_threshold_sec * 1000
            
            if closed_duration >= drowsy_threshold_ms:
                new_state = 'Drowsy'
            elif closed_duration >= (Config.BLINK_TIME_THRESHOLD * 1000 * 4): 
                new_state = 'Fatigued'
            else:
                new_state = self.current_state 
                if self.current_state == 'Drowsy': 
                     new_state = 'Drowsy'
                else:
                     new_state = 'Alert' 
        else:
            if self.closed_start_time is not None:
                closed_duration = timestamp_ms - self.closed_start_time
                blink_threshold_ms = Config.BLINK_TIME_THRESHOLD * 1000
                
                if closed_duration <= blink_threshold_ms * 2: 
                    self.blink_counter += 1
                    self.blink_history.append((timestamp_ms, 1))
                else:
                    self.blink_history.append((timestamp_ms, 0))
                
                self.closed_start_time = None
            else:
                self.blink_history.append((timestamp_ms, 0)) 
        
            new_state = 'Alert'

        window_ms = Config.WINDOW_TIME_SEC * 1000
        current_cutoff = timestamp_ms - window_ms
        
        while self.blink_history and self.blink_history[0][0] < current_cutoff:
            self.blink_history.popleft()
            
        recent_blinks = sum(1 for t, is_blink in self.blink_history if is_blink)
        
        if timestamp_ms > window_ms:
             
             bpm = recent_blinks / (Config.WINDOW_TIME_SEC / 60.0)
             
             if new_state == 'Alert' and bpm < Config.BLINK_RATE_DROWSY_THRESHOLD:
                 new_state = 'Fatigued'
        
        self.current_state = new_state
        return self.current_state
    
    def process_frame(self, frame, timestamp_ms=None):
       
        if timestamp_ms is None:
            import time
            timestamp_ms = time.time() * 1000
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        result = self.landmarker.detect(mp_image)
        
        if not result.face_landmarks:
    
            return {
                'face_detected': False,
                'ear': None,
                'state': self.current_state,
                'warning': (self.current_state == 'Drowsy')
            }
        
        face_landmarks = result.face_landmarks[0]
        
        left_eye, right_eye = self.extract_eye_landmarks(face_landmarks, frame.shape)
        
        left_ear = self.calculate_ear(left_eye)
        right_ear = self.calculate_ear(right_eye)
        ear = (left_ear + right_ear) / 2.0
        
        self.ear_history.append(ear) 
        state = self.update_state(ear, timestamp_ms)
        warning = (state == 'Drowsy')
        self.frame_counter += 1
        
        closed_duration_sec = 0.0
        if self.closed_start_time is not None:
            closed_duration_sec = (timestamp_ms - self.closed_start_time) / 1000.0
        
        return {
            'face_detected': True,
            'ear': ear,
            'left_eye': left_eye,
            'right_eye': right_eye,
            'state': state,
            'warning': warning,
            'closed_duration_sec': closed_duration_sec, 
            'closed_eye_frames': int(closed_duration_sec * 30),
            'blink_count': self.blink_counter
        }
    
    def reset(self):
        self.ear_history.clear()
        self.blink_history.clear()
        self.frame_counter = 0
        self.closed_eye_counter = 0
        self.blink_counter = 0
        self.current_state = 'Alert'
        self.state_counter = 0
