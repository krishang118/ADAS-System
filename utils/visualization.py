import cv2
import numpy as np

class Visualizer:
    
    @staticmethod
    def draw_lane_mask(frame, mask, color=(0, 255, 0), alpha=0.3):
    
        overlay = frame.copy()
        overlay[mask > 0] = color
        return cv2.addWeighted(frame, 1-alpha, overlay, alpha, 0)
    
    @staticmethod
    def draw_lane_lines(frame, left_fit, right_fit, color=(0, 255, 0), thickness=5):
        
        if left_fit is None and right_fit is None:
            return frame
        
        h, w = frame.shape[:2]
        y_values = np.linspace(0, h-1, h)
        
        if left_fit is not None:
            left_x = left_fit[0]*y_values**2 + left_fit[1]*y_values + left_fit[2]
            left_points = np.column_stack((left_x, y_values)).astype(np.int32)
            cv2.polylines(frame, [left_points], False, color, thickness)
        
        if right_fit is not None:
            right_x = right_fit[0]*y_values**2 + right_fit[1]*y_values + right_fit[2]
            right_points = np.column_stack((right_x, y_values)).astype(np.int32)
            cv2.polylines(frame, [right_points], False, color, thickness)
        
        return frame
    
    @staticmethod
    def draw_boxes(frame, detections, color=(0, 255, 0), thickness=2, show_conf=True):
    
        for det in detections:
            if isinstance(det, dict):
                bbox = det['bbox']
                conf = det.get('conf', None)
            else:
                bbox = det[:4]
                conf = det[4] if len(det) > 4 else None
            
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            if show_conf and conf is not None:
                label = f"{conf:.2f}"
                cv2.putText(frame, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame
    
    @staticmethod
    def draw_warning(frame, text, position='top', color=(0, 0, 255)):
    
        h, w = frame.shape[:2]
        if position == 'top':
            pos = (w//2 - 100, 50)
        elif position == 'bottom':
            pos = (w//2 - 100, h - 50)
        else:
            pos = position
        
        cv2.rectangle(frame, (pos[0]-10, pos[1]-30), (pos[0]+210, pos[1]+10), (0, 0, 0), -1)
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        return frame
    
    @staticmethod
    def draw_info_panel(frame, info_dict, position='top-left'):
       
        h, w = frame.shape[:2]
        if position == 'top-left':
            start_x, start_y = 10, 30
        elif position == 'top-right':
            start_x, start_y = w - 200, 30
        else:
            start_x, start_y = position
        
        y_offset = 0
        for key, value in info_dict.items():
            text = f"{key}: {value}"
            cv2.putText(frame, text, (start_x, start_y + y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
        
        return frame
