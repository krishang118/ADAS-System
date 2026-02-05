import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import time
import tempfile
import sys
sys.path.append(str(Path(__file__).parent))
from config.config import Config
from inference.lane_detector import LaneDetector
from inference.fcw_detector import FCWDetector
from inference.pedestrian_detector import PedestrianDetector
from inference.sign_detector import TrafficSignDetector
from inference.light_detector import TrafficLightDetector
from inference.twowheeler_detector import TwoWheelerDetector
from inference.animal_detector import AnimalDetector
from inference.drowsiness_monitor import DrowsinessMonitor
from utils.visualization import Visualizer

st.set_page_config(
    page_title="The ADAS System",
    page_icon="ADAS",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #000000 0%, #0a0a0a 100%);
        background-attachment: fixed;
    }
    
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(2px 2px at 20% 30%, rgba(255, 255, 255, 0.03), transparent),
            radial-gradient(2px 2px at 60% 70%, rgba(255, 255, 255, 0.03), transparent),
            radial-gradient(1px 1px at 50% 50%, rgba(255, 255, 255, 0.02), transparent);
        background-size: 200% 200%;
       background-position: 0% 0%;
        animation: drift 20s ease infinite;
        pointer-events: none;
        z-index: 0;
    }
    
    @keyframes drift {
        0%, 100% { background-position: 0% 0%; }
        50% { background-position: 100% 100%; }
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #e0e0e0 !important;
        font-weight: 600;
    }
    
    h1 {
        font-weight: 700;
        letter-spacing: -1px;
    }
    
    p, li, label, div {
        color: #b0b0b0 !important;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0a0a 0%, #000000 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    [data-testid="stSidebar"] * {
        color: #b0b0b0 !important;
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #e0e0e0 !important;
    }
    
    .stButton button {
        background: linear-gradient(135deg, #1a1a1a 0%, #0f0f0f 100%);
        color: white !important;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 12px 28px;
        font-weight: 600;
        font-size: 14px;
        letter-spacing: 0.5px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(255, 255, 255, 0.1);
        border-color: rgba(255, 255, 255, 0.2);
    }
    
    .stTextInput > div[data-baseweb="input"] {
        border-color: rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
    }
    
    .stTextInput > div[data-baseweb="input"]:focus-within {
        border-color: rgba(255, 255, 255, 0.5) !important;
        box-shadow: 0 0 10px rgba(255, 255, 255, 0.1) !important;
    }

    .stTextInput input {
        background: rgba(20, 20, 20, 0.8) !important;
        color: #FAFAFA !important;
        padding: 14px 20px !important;
        font-size: 15px !important;
    }
    
    .stSelectbox > div > div {
        background: rgba(20, 20, 20, 0.8) !important;
        border: 2px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
    }
    
    .stRadio > div {
        background: rgba(20, 20, 20, 0.4) !important;
        border-radius: 12px !important;
        padding: 10px !important;
    }
    
    .stCheckbox {
        color: #b0b0b0 !important;
    }
    
    .st Slider > div > div {
        color: #e0e0e0 !important;
    }
    
    .stFileUploader > div {
        background: rgba(20, 20, 20, 0.6) !important;
        border: 2px dashed rgba(255, 255, 255, 0.2) !important;
        border-radius: 12px !important;
        padding: 20px !important;
    }
    
    .stSuccess {
        background: linear-gradient(135deg, rgba(40, 167, 69, 0.15) 0%, rgba(34, 139, 58, 0.15) 100%) !important;
        border: 1px solid rgba(40, 167, 69, 0.3) !important;
        border-radius: 8px !important;
        color: #28a745 !important;
    }
    
    .stWarning {
        background: linear-gradient(135deg, rgba(255, 193, 7, 0.15) 0%, rgba(255, 152, 0, 0.15) 100%) !important;
        border: 1px solid rgba(255, 193, 7, 0.3) !important;
        border-radius: 8px !important;
        color: #ffc107 !important;
    }
    
    .stError {
        background: linear-gradient(135deg, rgba(220, 53, 69, 0.15) 0%, rgba(176, 42, 55, 0.15) 100%) !important;
        border: 1px solid rgba(220, 53, 69, 0.3) !important;
        border-radius: 8px !important;
        color: #dc3545 !important;
    }
    
    .stInfo {
        background: linear-gradient(135deg, rgba(23, 162, 184, 0.15) 0%, rgba(19, 132, 150, 0.15) 100%) !important;
        border: 1px solid rgba(23, 162, 184, 0.3) !important;
        border-radius: 8px !important;
        color: #17a2b8 !important;
    }
    
    [data-testid="stMetricValue"] {
        color: #e0e0e0 !important;
        font-size: 28px !important;
        font-weight: 600 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #b0b0b0 !important;
    }
    
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        margin: 30px 0;
    }
    
    code {
        background: rgba(255, 255, 255, 0.05) !important;
        color: #cccccc !important;
        padding: 3px 8px;
        border-radius: 6px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stSpinner > div {
        border-top-color: #ffffff !important;
    }
    
    .stImage {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
</style>
""", unsafe_allow_html=True)

if 'detector' not in st.session_state:
    st.session_state.detector = None
if 'detectors' not in st.session_state:
    st.session_state.detectors = {}  
if 'running' not in st.session_state:
    st.session_state.running = False
if 'current_module' not in st.session_state:
    st.session_state.current_module = None
if 'enabled_modules' not in st.session_state:
    st.session_state.enabled_modules = set()

st.markdown("""
    <div style='text-align: center; padding-bottom: 20px;'>
        <h1 style='
            background: linear-gradient(to right, #e0e0e0, #a0a0a0);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.5rem;
            font-weight: 800;
            margin-bottom: 0;
        '>The Advanced Driver Assistance System (ADAS)</h1>
        <p style='color: #b0b0b0; font-size: 1.2rem; font-weight: 400;'>
            It watches, you drive.
        </p>
    </div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## System Configuration")
        
    st.markdown("---")

    st.subheader("Input Source")
    input_source = st.radio(
        "Select Input Source",
        ["Upload Video", "Phone Camera (IP Cam)", "Webcam"],
        label_visibility="collapsed"
    )

st.sidebar.subheader("ADAS Module")
adas_module = st.sidebar.selectbox(
    "Select Module",
    [
        "Lane Detection + LDW",
        "Forward Collision Warning",
        "Pedestrian Detection",
        "Two-Wheeler Detection",
        "Animal Awareness",
        "Traffic Sign Recognition",
        "Traffic Light Detection",
        "Driver Drowsiness Monitor",
        "The ADAS System"
    ]
)

st.sidebar.subheader("Parameters")

if adas_module == "Forward Collision Warning":
    ttc_threshold = st.sidebar.slider("TTC Threshold (s)", 0.5, 5.0, Config.FCW_TTC_THRESHOLD, 0.1)
    fcw_conf = st.sidebar.slider("Confidence", 0.1, 1.0, Config.FCW_CONF_THRESHOLD, 0.05)
elif adas_module == "Pedestrian Detection":
    ped_conf = st.sidebar.slider("Confidence", 0.1, 1.0, Config.PED_CONF_THRESHOLD, 0.05)
    ped_distance = st.sidebar.slider("Warning Distance (px)", 20, 300, Config.PED_DISTANCE_THRESHOLD, 10)
elif adas_module == "Traffic Sign Recognition":
    sign_conf = st.sidebar.slider("Confidence", 0.1, 1.0, Config.SIGN_CONF_THRESHOLD, 0.05)
elif adas_module == "Traffic Light Detection":
    light_conf = st.sidebar.slider("Confidence", 0.1, 1.0, Config.LIGHT_CONF_THRESHOLD, 0.05)
elif adas_module == "Lane Detection + LDW":
    ldw_threshold = st.sidebar.slider("LDW Offset Threshold", 0.1, 0.5, Config.LDW_OFFSET_THRESHOLD, 0.05)
elif adas_module == "Two-Wheeler Detection":
    twowheeler_conf = st.sidebar.slider("Confidence", 0.1, 1.0, Config.TWOWHEELER_CONF_THRESHOLD, 0.05)
    twowheeler_distance = st.sidebar.slider("Warning Distance (px)", 20, 200, Config.TWOWHEELER_DISTANCE_THRESHOLD, 10)
elif adas_module == "Animal Awareness":
    animal_conf = st.sidebar.slider("Confidence", 0.1, 1.0, Config.ANIMAL_CONF_THRESHOLD, 0.05)
    animal_distance = st.sidebar.slider("Warning Distance (px)", 20, 150, Config.ANIMAL_DISTANCE_THRESHOLD, 10)
elif adas_module == "Driver Drowsiness Monitor":
    st.sidebar.markdown("**EAR Thresholds**")
    ear_alert = st.sidebar.slider("EAR Alert Threshold", 0.15, 0.35, Config.EAR_THRESHOLD_ALERT, 0.01)
    ear_drowsy = st.sidebar.slider("EAR Drowsy Threshold", 0.10, 0.25, Config.EAR_THRESHOLD_DROWSY, 0.01)
    st.sidebar.markdown("**Temporal Settings**")
    drowsy_time = st.sidebar.slider("Drowsy Duration (seconds)", 0.5, 3.0, Config.DROWSY_TIME_THRESHOLD, 0.1)
    drowsy_frames = None 
    show_landmarks = st.sidebar.checkbox("Show Eye Landmarks", value=True)
elif adas_module == "The ADAS System":
    st.sidebar.markdown("**Enabled Components**")
    enable_lane = st.sidebar.checkbox("Lane Detection + LDW", value=True)
    enable_fcw = st.sidebar.checkbox("Forward Collision Warning", value=True)
    enable_ped = st.sidebar.checkbox("Pedestrian Detection", value=True)
    enable_two = st.sidebar.checkbox("Two-Wheeler Detection", value=True)
    enable_animal = st.sidebar.checkbox("Animal Awareness", value=True)
    enable_signs = st.sidebar.checkbox("Traffic Sign Recognition", value=True)
    enable_lights = st.sidebar.checkbox("Traffic Light Detection", value=True)    
    ldw_threshold = Config.LDW_OFFSET_THRESHOLD
    ttc_threshold = Config.FCW_TTC_THRESHOLD
    fcw_conf = Config.FCW_CONF_THRESHOLD
    ped_conf = Config.PED_CONF_THRESHOLD
    ped_distance = Config.PED_DISTANCE_THRESHOLD
    sign_conf = Config.SIGN_CONF_THRESHOLD
    light_conf = Config.LIGHT_CONF_THRESHOLD
    twowheeler_conf = Config.TWOWHEELER_CONF_THRESHOLD
    twowheeler_distance = Config.TWOWHEELER_DISTANCE_THRESHOLD
    animal_conf = Config.ANIMAL_CONF_THRESHOLD
    animal_distance = Config.ANIMAL_DISTANCE_THRESHOLD
    ear_alert = Config.EAR_THRESHOLD_ALERT
    ear_drowsy = Config.EAR_THRESHOLD_DROWSY
    drowsy_time = Config.DROWSY_TIME_THRESHOLD
    drowsy_frames = None

frame_skip = st.sidebar.slider("Frame Skip (process every Nth frame)", 1, 10, 2)

fps_override = st.sidebar.slider("Target FPS Limit (0 = Auto/Video Source)", 0, 60, 0, 5)

st.sidebar.subheader("Display Options")
show_fps = st.sidebar.checkbox("Show FPS", value=True)
show_warnings = st.sidebar.checkbox("Show Warnings", value=True)
high_quality = st.sidebar.checkbox("High Quality Mode (Slower)", value=False, help="Runs inference at 1280p for better small object detection")

phone_url = None
if input_source == "Phone Camera (IP Cam)":
    st.sidebar.subheader("Phone Camera URL")
    phone_url = st.sidebar.text_input("URL", Config.PHONE_CAMERA_URL, label_visibility="collapsed")

uploaded_file = None
if input_source == "Upload Video":
    uploaded_file = st.sidebar.file_uploader("Upload Video File", type=['mp4', 'avi', 'mov'])

col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("### Video Feed")
    st.markdown('<div class="video-container">', unsafe_allow_html=True)
    video_placeholder = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown("### Live Analytics")
    
    status_placeholder = st.empty()
    
    st.markdown("---")
    
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        fps_placeholder = st.empty()
    with m_col2:
        stats_placeholder = st.empty()
        
    st.markdown("---")
    
    warning_placeholder = st.empty()

st.markdown("<br>", unsafe_allow_html=True)
c1, c2, c3, c4, c5 = st.columns([1, 2, 2, 2, 1])

with c2:
    start_button = st.button("Start System", use_container_width=True)

with c3:
    stop_button = st.button("Stop System", type="primary", use_container_width=True)

with c4:
    reset_button = st.button("Reset State", use_container_width=True)

def initialize_detector(module_name):
    try:
        if module_name == "Lane Detection + LDW":
            return LaneDetector()
        elif module_name == "Forward Collision Warning":
            return FCWDetector()
        elif module_name == "Pedestrian Detection":
            return PedestrianDetector()
        elif module_name == "Traffic Sign Recognition":
            return TrafficSignDetector()
        elif module_name == "Traffic Light Detection":
            return TrafficLightDetector()
        elif module_name == "Two-Wheeler Detection":
            return TwoWheelerDetector()
        elif module_name == "Animal Awareness":
            return AnimalDetector()
        elif module_name == "Driver Drowsiness Monitor":
            return DrowsinessMonitor()
    except Exception as e:
        st.error(f"Error initializing detector: {e}")
        return None

def process_frame(frame, detector, module_name, conf_threshold=None, param_threshold=None, imgsz=640,
                  ear_alert=None, ear_drowsy=None, drowsy_frames=None, drowsy_time=None, fps=30.0, timestamp_ms=None):
    warning_text = None
    info = {}
    
    try:
        if module_name == "Lane Detection + LDW":
            mask = detector.detect(frame)
            frame = Visualizer.draw_lane_mask(frame, mask)
            
            left_fit, right_fit = detector.fit_lanes(mask)
            
            offset = detector.calculate_offset(left_fit, right_fit, frame.shape[0], frame.shape[1])
            
            warning, direction = detector.check_departure_warning(offset, threshold=param_threshold)
            
            if offset is not None:
                info['Lane Offset'] = f"{offset:.3f}"
            
            if warning:
                warning_text = f"LANE DEPARTURE - {direction.upper()}"
                frame = Visualizer.draw_warning(frame, warning_text)
        
        elif module_name == "Forward Collision Warning":
            
            detections = detector.detect(frame, conf=conf_threshold)
            
            tracked = detector.track(detections)
            
            ttc_thresh = param_threshold if param_threshold is not None else Config.FCW_TTC_THRESHOLD
            ttc_results = detector.calculate_ttc(tracked, frame.shape[0])
            
            ttc_warnings = []
            for item in ttc_results:
                is_warning = item['ttc'] < ttc_thresh
                item['warning'] = is_warning 
                ttc_warnings.append(item)
            
            for warn_item in ttc_warnings:
                color = (0, 0, 255) if warn_item['warning'] else (0, 255, 0)
                bbox = warn_item['bbox']
                
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                ttc_val = warn_item['ttc']
                if ttc_val != float('inf'):
                    label = f"TTC: {ttc_val:.1f}s"
                    cv2.putText(frame, label, (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                if warn_item['warning']:
                    warning_text = f"COLLISION WARNING - TTC: {ttc_val:.1f}s"
            
            info['Vehicles'] = len(ttc_warnings)
            
            if warning_text:
                frame = Visualizer.draw_warning(frame, warning_text)
        
        elif module_name == "Pedestrian Detection":
            detections = detector.detect(frame, conf_threshold=conf_threshold)
            
            ped_thresh = param_threshold if param_threshold is not None else Config.PED_DISTANCE_THRESHOLD
            
            warnings = []
            raw_warnings = detector.check_warning(detections, frame.shape[0], frame.shape[1])
            
            for i, det in enumerate(detections):
                x1, y1, x2, y2 = det['bbox']
                height = y2 - y1
                y_center = (y1 + y2) / 2
                
                is_warning = height > ped_thresh or y_center > frame.shape[0] * 0.5
                
                msg = raw_warnings[i]
                msg['warning'] = is_warning
                warnings.append(msg)
            
            for warn_item in warnings:
                color = (0, 0, 255) if warn_item['warning'] else (0, 255, 0)
                bbox = warn_item['bbox']
                
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                if warn_item['warning']:
                    warning_text = "PEDESTRIAN DETECTED"
            
            info['Pedestrians'] = len(detections)
            
            if warning_text:
                frame = Visualizer.draw_warning(frame, warning_text)
        
        elif module_name == "Traffic Sign Recognition":
            if hasattr(detector, 'detect_with_size'): 
                 detections = detector.detect(frame, conf_threshold=conf_threshold, imgsz=imgsz)
            else:
                 detections = detector.detect(frame, conf_threshold=conf_threshold) 
            
            for det in detections:
                x1, y1, x2, y2 = map(int, det['bbox'])
                color = (255, 0, 0)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                label = f"{det['class']}: {det['conf']:.2f}"
                cv2.putText(frame, label, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            info['Signs Detected'] = len(detections)
        
        elif module_name == "Traffic Light Detection":
            detections = detector.detect(frame, conf_threshold=conf_threshold, imgsz=imgsz)
            
            for det in detections:
                x1, y1, x2, y2 = map(int, det['bbox'])
                state = det['state']
                
                if state == 'red':
                    color = (0, 0, 255)
                    warning_text = "RED LIGHT"
                elif state == 'yellow':
                    color = (0, 255, 255)
                elif state == 'green':
                    color = (0, 255, 0)
                else:
                    color = (128, 128, 128)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                label = f"{state.upper()}: {det['conf']:.2f}"
                cv2.putText(frame, label, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            info['Lights Detected'] = len(detections)
            
            if warning_text:
                frame = Visualizer.draw_warning(frame, warning_text)
        
        elif module_name == "Two-Wheeler Detection":

            detections = detector.detect(frame, conf_threshold=conf_threshold)
            
            twowheeler_thresh = param_threshold if param_threshold is not None else Config.TWOWHEELER_DISTANCE_THRESHOLD
            
            warnings = []
            raw_warnings = detector.check_warning(detections, frame.shape[0], frame.shape[1])
            
            for i, det in enumerate(detections):
                x1, y1, x2, y2 = det['bbox']
                height = y2 - y1
                y_center = (y1 + y2) / 2
                
                is_warning = height > twowheeler_thresh or y_center > frame.shape[0] * 0.4
                
                msg = raw_warnings[i]
                msg['warning'] = is_warning
                warnings.append(msg)
            
            for warn_item in warnings:
                color = (0, 0, 255) if warn_item['warning'] else (0, 255, 0)
                bbox = warn_item['bbox']
                
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                label = f"{warn_item['type']}: {warn_item['conf']:.2f}"
                cv2.putText(frame, label, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                if warn_item['warning']:
                    warning_text = f"{warn_item['type'].upper()} DETECTED"
            
            info['Two-Wheelers'] = len(detections)
            
            if warning_text:
                frame = Visualizer.draw_warning(frame, warning_text)
        
        elif module_name == "Animal Awareness":
            detections = detector.detect(frame, conf_threshold=conf_threshold)
            
            animal_thresh = param_threshold if param_threshold is not None else Config.ANIMAL_DISTANCE_THRESHOLD
            
            warnings = []
            raw_warnings = detector.check_warning(detections, frame.shape[0], frame.shape[1])
            
            for i, det in enumerate(detections):
                x1, y1, x2, y2 = det['bbox']
                height = y2 - y1
                y_center = (y1 + y2) / 2
                
                is_warning = height > animal_thresh or y_center > frame.shape[0] * 0.5
                
                msg = raw_warnings[i]
                msg['warning'] = is_warning
                warnings.append(msg)
            
            for warn_item in warnings:
                color = (255, 140, 0) if warn_item['warning'] else (0, 255, 0)  
                bbox = warn_item['bbox']
                
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                label = f"{warn_item['type']}: {warn_item['conf']:.2f}"
                cv2.putText(frame, label, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                if warn_item['warning']:
                    warning_text = f"ANIMAL HAZARD - {warn_item['type'].upper()}"
            
            info['Animals'] = len(detections)
            
            if warning_text:
                frame = Visualizer.draw_warning(frame, warning_text)
        
        elif module_name == "Driver Drowsiness Monitor":
            result = detector.detect(frame, ear_alert=ear_alert, ear_drowsy=ear_drowsy, 
                                   drowsy_frames=drowsy_frames, drowsy_time_sec=drowsy_time, fps=fps, timestamp_ms=timestamp_ms)
            
            frame = detector.draw_visualization(frame, result)
            
            if result['face_detected']:
                info['Driver State'] = result['state']
                if result['ear'] is not None:
                    info['EAR'] = f"{result['ear']:.3f}"
                info['Blink Count'] = result.get('blink_count', 0)
                
                if result['warning']:
                    warning_text = "DRIVER DROWSINESS DETECTED"
            else:
                info['Status'] = 'No face detected'
    
    except Exception as e:
        st.error(f"Error processing frame: {e}")
    
    return frame, warning_text, info

def process_frame_combined(frame, detectors, enabled_modules, conf_params, param_params, fps=30.0):
    warnings = []
    info = {}
    
    try:
        if "Lane Detection + LDW" in enabled_modules:
            detector = detectors["Lane Detection + LDW"]
            mask = detector.detect(frame)
            frame = Visualizer.draw_lane_mask(frame, mask)
            left_fit, right_fit = detector.fit_lanes(mask)
            
            offset = detector.calculate_offset(left_fit, right_fit, frame.shape[0], frame.shape[1])
            warning, direction = detector.check_departure_warning(offset, threshold=param_params.get('ldw_threshold'))
            
            if offset is not None:
                info['Lane Offset'] = f"{offset:.3f}"
            if warning:
                warnings.append(f"LANE DEPARTURE - {direction.upper()}")
        
        if "Forward Collision Warning" in enabled_modules:
            detector = detectors["Forward Collision Warning"]
            detections = detector.detect(frame, conf=conf_params.get('fcw_conf'))
            tracked = detector.track(detections)
            ttc_results = detector.calculate_ttc(tracked, frame.shape[0])
            
            ttc_thresh = param_params.get('ttc_threshold')
            for item in ttc_results:
                is_warning = item['ttc'] < ttc_thresh
                item['warning'] = is_warning
                
                color = (0, 0, 255) if is_warning else (0, 255, 0)
                bbox = item['bbox']
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                ttc_val = item['ttc']
                if ttc_val != float('inf'):
                    label = f"TTC: {ttc_val:.1f}s"
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                if is_warning:
                    warnings.append(f"COLLISION WARNING - TTC: {ttc_val:.1f}s")
            
            info['Vehicles'] = len(ttc_results)
        
        if "Pedestrian Detection" in enabled_modules:
            detector = detectors["Pedestrian Detection"]
            detections = detector.detect(frame, conf_threshold=conf_params.get('ped_conf'))
            
            ped_thresh = param_params.get('ped_distance')
            raw_warnings = detector.check_warning(detections, frame.shape[0], frame.shape[1])
            
            for i, det in enumerate(detections):
                x1, y1, x2, y2 = det['bbox']
                height = y2 - y1
                y_center = (y1 + y2) / 2
                
                is_warning =  height > ped_thresh or y_center > frame.shape[0] * 0.5
                color = (0, 0, 255) if is_warning else (0, 255, 0)
                
                x1, y1, x2, y2 = map(int, det['bbox'])
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                if is_warning:
                    warnings.append("PEDESTRIAN DETECTED")
            
            info['Pedestrians'] = len(detections)
        
        if "Two-Wheeler Detection" in enabled_modules:
            detector = detectors["Two-Wheeler Detection"]
            detections = detector.detect(frame, conf_threshold=conf_params.get('twowheeler_conf'))
            
            twowheeler_thresh = param_params.get('twowheeler_distance')
            raw_warnings = detector.check_warning(detections, frame.shape[0], frame.shape[1])
            
            for i, det in enumerate(detections):
                x1, y1, x2, y2 = det['bbox']
                height = y2 - y1
                y_center = (y1 + y2) / 2
                
                is_warning = height > twowheeler_thresh or y_center > frame.shape[0] * 0.4
                
                msg = raw_warnings[i]
                color = (0, 0, 255) if is_warning else (0, 255, 0)
                bbox = msg['bbox']
                
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                label = f"{msg['type']}: {msg['conf']:.2f}"
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                if is_warning:
                    warnings.append(f"{msg['type'].upper()} DETECTED")
            
            info['Two-Wheelers'] = len(detections)
        
        if "Animal Awareness" in enabled_modules:
            detector = detectors["Animal Awareness"]
            detections = detector.detect(frame, conf_threshold=conf_params.get('animal_conf'))
            
            animal_thresh = param_params.get('animal_distance')
            raw_warnings = detector.check_warning(detections, frame.shape[0], frame.shape[1])
            
            for i, det in enumerate(detections):
                x1, y1, x2, y2 = det['bbox']
                height = y2 - y1
                y_center = (y1 + y2) / 2
                
                is_warning = height > animal_thresh or y_center > frame.shape[0] * 0.5
                
                msg = raw_warnings[i]
                color = (255, 140, 0) if is_warning else (0, 255, 0)  
                bbox = msg['bbox']
                
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                label = f"{msg['type']}: {msg['conf']:.2f}"
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                if is_warning:
                    warnings.append(f"ANIMAL HAZARD - {msg['type'].upper()}")
            
            info['Animals'] = len(detections)
        
        if "Traffic Sign Recognition" in enabled_modules:
            detector = detectors["Traffic Sign Recognition"]
            detections = detector.detect(frame, conf_threshold=conf_params.get('sign_conf'))
            
            for det in detections:
                x1, y1, x2, y2 = map(int, det['bbox'])
                color = (255, 0, 0)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                label = f"{det['class']}: {det['conf']:.2f}"
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            info['Signs Detected'] = len(detections)
        
        if "Traffic Light Detection" in enabled_modules:
            detector = detectors["Traffic Light Detection"]
            detections = detector.detect(frame, conf_threshold=conf_params.get('light_conf'), imgsz=640)
            
            for det in detections:
                x1, y1, x2, y2 = map(int, det['bbox'])
                state = det['state']
                
                if state == 'red':
                    color = (0, 0, 255)
                    warnings.append("RED LIGHT")
                elif state == 'yellow':
                    color = (0, 255, 255)
                elif state == 'green':
                    color = (0, 255, 0)
                else:
                    color = (128, 128, 128)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                label = f"{state.upper()}: {det['conf']:.2f}"
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            info['Lights Detected'] = len(detections)
        
    except Exception as e:
        st.error(f"Error in combined processing: {e}")
    unique_warnings = list(dict.fromkeys(warnings))
    combined_warning = " | ".join(unique_warnings) if unique_warnings else None
    
    return frame, combined_warning, info
if start_button:
    st.session_state.running = True
    
    if adas_module == "The ADAS System":
        enabled_modules = []
        if enable_lane: enabled_modules.append("Lane Detection + LDW")
        if enable_fcw: enabled_modules.append("Forward Collision Warning")
        if enable_ped: enabled_modules.append("Pedestrian Detection")
        if enable_two: enabled_modules.append("Two-Wheeler Detection")
        if enable_animal: enabled_modules.append("Animal Awareness")
        if enable_signs: enabled_modules.append("Traffic Sign Recognition")
        if enable_lights: enabled_modules.append("Traffic Light Detection")
        
        st.session_state.enabled_modules = set(enabled_modules)
        
        for module in enabled_modules:
            if module not in st.session_state.detectors or reset_button:
                st.session_state.detectors[module] = initialize_detector(module)
        
        st.session_state.current_module = adas_module
        
    else:
        if st.session_state.detector is None or reset_button:
            st.session_state.detector = initialize_detector(adas_module)
            st.session_state.current_module = adas_module  
        
        if st.session_state.detector is None:
            st.error("Failed to initialize detector. Please check model weights.")
            st.session_state.running = False

any_detector_active = (st.session_state.detector is not None) or (len(st.session_state.detectors) > 0)

if any_detector_active:
    module_changed = st.session_state.current_module != adas_module
    
    checkboxes_changed = False
    if adas_module == "The ADAS System" and st.session_state.current_module == "The ADAS System":
        current_selection = set()
        if enable_lane: current_selection.add("Lane Detection + LDW")
        if enable_fcw: current_selection.add("Forward Collision Warning")
        if enable_ped: current_selection.add("Pedestrian Detection")
        if enable_two: current_selection.add("Two-Wheeler Detection")
        if enable_animal: current_selection.add("Animal Awareness")
        if enable_signs: current_selection.add("Traffic Sign Recognition")
        if enable_lights: current_selection.add("Traffic Light Detection")
        
        checkboxes_changed = current_selection != st.session_state.enabled_modules
    
    if module_changed:
        st.info(f"Module changed from **{st.session_state.current_module}** to **{adas_module}**. Reinitializing detector...")
        st.session_state.detector = None
        st.session_state.detectors = {}
        st.session_state.current_module = None 
        st.session_state.running = False  
    elif checkboxes_changed:
        st.info("Component selection changed. Reinitializing detectors...")
        st.session_state.detectors = {}
        st.session_state.running = False  
        temp_module = st.session_state.current_module
        st.session_state.current_module = None

if stop_button or reset_button:
    st.session_state.running = False
    if reset_button:
        st.session_state.detector = None
        st.session_state.detectors = {}

has_detectors = (st.session_state.detector is not None) or (len(st.session_state.detectors) > 0)

if st.session_state.running and has_detectors:
    cap = None
    
    try:
        if input_source == "Upload Video" and uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            cap = cv2.VideoCapture(tfile.name)
        
        elif input_source == "Phone Camera (IP Cam)" and phone_url:
            cap = cv2.VideoCapture(phone_url)
        
        elif input_source == "Webcam":
            cap = cv2.VideoCapture(0)
        
        if cap is None or not cap.isOpened():
            st.error("Failed to open video source")
            st.session_state.running = False
        else:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0 or np.isnan(fps):
                fps = 30.0 
            
            st.info(f"Video FPS: {fps:.2f} (Syncing playback)")
            
            frame_count = 0
            fps_list = []
            prev_frame_time = time.time()
            sync_start_time = time.time() 
            
            while st.session_state.running:
                loop_start = time.time()
                
                dt = loop_start - prev_frame_time
                prev_frame_time = loop_start
                
                current_fps = 1.0 / dt if dt > 0 else 0
                if len(fps_list) > 30:
                    fps_list.pop(0)
                fps_list.append(current_fps)
                avg_fps = np.mean(fps_list) if fps_list else 0
                
                ret, frame = cap.read()
                
                if not ret:
                    st.warning("End of video or camera disconnected")
                    break
                
                frame_count += 1
                should_process = (frame_count % frame_skip == 0)
                
                if should_process:
                    img_size = 1280 if high_quality else 640
                    current_conf = None
                    current_param = None
                    
                    if adas_module == "Forward Collision Warning":
                        current_conf = fcw_conf
                        current_param = ttc_threshold
                    elif adas_module == "Pedestrian Detection":
                        current_conf = ped_conf
                        current_param = ped_distance
                    elif adas_module == "Traffic Sign Recognition":
                        current_conf = sign_conf
                    elif adas_module == "Traffic Light Detection":
                        current_conf = light_conf
                    elif adas_module == "Lane Detection + LDW":
                        current_param = ldw_threshold
                    elif adas_module == "Two-Wheeler Detection":
                        current_conf = twowheeler_conf
                        current_param = twowheeler_distance
                    elif adas_module == "Animal Awareness":
                        current_conf = animal_conf
                        current_param = animal_distance
                    if adas_module == "Driver Drowsiness Monitor" or adas_module == "The ADAS System":
                        if input_source == "Upload Video" or input_source == "Phone Camera (IP Cam)":
                             logic_fps = fps if fps > 0 else 30.0
                        else:
                             logic_fps = avg_fps if avg_fps > 0 else 30.0
                        
                        drowsy_frames = max(1, int(drowsy_time * logic_fps))
                    else:
                        logic_fps = avg_fps if avg_fps > 0 else 30.0
                                        
                    if adas_module == "The ADAS System":
                        conf_params = {
                            'fcw_conf': fcw_conf,
                            'ped_conf': ped_conf,
                            'sign_conf': sign_conf,
                            'light_conf': light_conf,
                            'twowheeler_conf': twowheeler_conf,
                            'animal_conf': animal_conf
                        }
                        param_params = {
                            'ldw_threshold': ldw_threshold,
                            'ttc_threshold': ttc_threshold,
                            'ped_distance': ped_distance,
                            'twowheeler_distance': twowheeler_distance,
                            'animal_distance': animal_distance
                        }
                        
                        processed_frame, warning_text, info = process_frame_combined(
                            frame,
                            st.session_state.detectors,
                            st.session_state.enabled_modules,
                            conf_params,
                            param_params,
                            fps=logic_fps
                        )
                    else:
                        processed_frame, warning_text, info = process_frame(
                            frame, 
                            st.session_state.detector, 
                            adas_module,
                            conf_threshold=current_conf,
                            param_threshold=current_param,
                            imgsz=img_size,
                            ear_alert=ear_alert if adas_module == "Driver Drowsiness Monitor" else None,
                            ear_drowsy=ear_drowsy if adas_module == "Driver Drowsiness Monitor" else None,
                            drowsy_frames=drowsy_frames if adas_module == "Driver Drowsiness Monitor" else None,
                            drowsy_time=drowsy_time if adas_module == "Driver Drowsiness Monitor" else None,
                            fps=logic_fps if adas_module == "Driver Drowsiness Monitor" else avg_fps
                        )
                    
                    if show_fps:
                        info['FPS'] = f"{avg_fps:.1f}"
                    
                    processed_frame = Visualizer.draw_info_panel(processed_frame, info)
                    
                    processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    
                    video_placeholder.image(processed_frame, channels="RGB", use_container_width=True)
                    
                    module_colors = {
                        "Lane Detection + LDW": "#17a2b8",
                        "Forward Collision Warning": "#dc3545",
                        "Pedestrian Detection": "#28a745",
                        "Traffic Sign Recognition": "#ffc107",
                        "Traffic Light Detection": "#ff9800",
                        "Two-Wheeler Detection": "#6f42c1",
                        "Animal Awareness": "#8B4513",
                        "Driver Drowsiness Monitor": "#00bcd4"
                    }
                    accent_color = module_colors.get(adas_module, "#ffffff")
                    
                    status_placeholder.markdown(f"""
                        <div style='
                            background: rgba(255, 255, 255, 0.05);
                            border-left: 4px solid {accent_color};
                            padding: 15px;
                            border-radius: 8px;
                            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                        '>
                            <div style='color: #b0b0b0; font-size: 0.9rem; margin-bottom: 5px;'>Active Module</div>
                            <div style='color: #e0e0e0; font-size: 1.1rem; font-weight: 600;'>{adas_module}</div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    if show_fps:
                        fps_placeholder.metric("Inference FPS", f"{avg_fps:.1f}")
                    
                    primary_stat_key = None
                    if "Vehicles" in info: primary_stat_key = "Vehicles"
                    elif "Pedestrians" in info: primary_stat_key = "Pedestrians"
                    elif "Signs Detected" in info: primary_stat_key = "Signs Detected"
                    elif "Lights Detected" in info: primary_stat_key = "Lights Detected"
                    elif "Two-Wheelers" in info: primary_stat_key = "Two-Wheelers"
                    elif "Animals" in info: primary_stat_key = "Animals"
                    elif "Lane Offset" in info: primary_stat_key = "Lane Offset"
                    elif "Driver State" in info: primary_stat_key = "Driver State"
                    
                    if primary_stat_key:
                        stats_placeholder.metric(primary_stat_key, info.get(primary_stat_key, "-"))
                    
                    if show_warnings and warning_text:
                        warning_placeholder.markdown(f"""
                            <div style='
                                background: linear-gradient(135deg, rgba(220, 53, 69, 0.2) 0%, rgba(176, 42, 55, 0.2) 100%);
                                border: 1px solid rgba(220, 53, 69, 0.4);
                                color: #ffcccc;
                                padding: 15px;
                                border-radius: 8px;
                                text-align: center;
                                font-weight: 600;
                                animation: pulse 2s infinite;
                            '>
                                {warning_text}
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        warning_placeholder.empty()
                
                loop_elapsed = time.time() - loop_start
                
                target_fps_val = fps_override if fps_override > 0 else fps
                if target_fps_val <= 0: target_fps_val = 30.0
                
                target_period = 1.0 / target_fps_val
                
                if loop_elapsed < target_period:
                    time.sleep(target_period - loop_elapsed)
            
            cap.release()
    
    except Exception as e:
        st.error(f"Error during processing: {e}")
    
    finally:
        if cap is not None:
            cap.release()

st.markdown("---")
st.markdown("""
<div style='margin-top: 20px;'>
    <h5 style='
        background: linear-gradient(to right, #b0b0b0, #808080);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 600;
        margin-bottom: 5px;
    '>"Safety is the first priority. No need to rush."</h5>
    <div style='text-align: right; color: #666666; font-style: italic; font-size: 0.9rem;'>
        ~ Dad
    </div>
</div>
""", unsafe_allow_html=True)
