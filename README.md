# The Advanced Driver Assistance System (ADAS)

A production-ready Advanced Driver Assistance System (ADAS) powered by computer vision and deep learning. Trained on ~160 GB of driving datasets, this system integrates 8 safety-critical components, from lane departure warnings to driver drowsiness monitoring, delivering real-time performance with fully local processing.

### Highlights

- 8 Production-Ready ADAS Components (Lane Detection, FCW, Pedestrian Detection, etc.)
- A Unified "The ADAS System" Mode, for running 7 road-based systems simultaneously
- Apple Silicon-optimized performance with MPS (Metal Performance Shaders)
- Fully Local Processing, no cloud dependencies after setup
- Interactive Streamlit UI with real-time running
- ~160 GB Training Data across multiple benchmark datasets

## Key Features

### 8 Complete ADAS Components

#### 1. Lane Detection + Lane Departure Warning (LDW)

- Model: UNet + ResNet18 Segmentation
- Dataset: CULane (~93 GB)
- Features:
  - Real-time lane boundary detection
  - Vehicle offset calculation from lane center
  - Configurable departure threshold warning

#### 2. Forward Collision Warning (FCW)

- Model: YOLOv8n (COCO 2017)
- Dataset: COCO 2017 (~25 GB)
- Features:
  - Vehicle detection (cars, trucks, buses)
  - SORT multi-object tracking
  - Time-To-Collision (TTC) calculation with configurable threshold

#### 3. Pedestrian Detection

- Model: YOLOv8n (COCO 2017)
- Dataset: COCO 2017 (~25 GB)
- Features:
  - Real-time pedestrian detection
  - Distance-based warning system
  - Height-based proximity estimation

#### 4. Two-Wheeler Detection

- Model: YOLOv8n (COCO 2017)
- Dataset: COCO 2017 (~25 GB)
- Features:
  - Detects bicycles and motorcycles
  - Separate alerts for different vehicle types

#### 5. Animal Awareness

- Model: YOLOv8n (COCO 2017)
- Dataset: COCO 2017 (~25 GB)
- Features:
  - Detects 6 animal classes (cat, dog, horse, sheep, cow, bear)
  - Distance-based warning system

#### 6. Traffic Sign Recognition

- Model: YOLOv8n (MTSD)
- Dataset: MTSD (~8.1 GB)
- Features:
  - Detects common traffic sign types
  - Real-time sign classification
  - Configurable confidence thresholds

#### 7. Traffic Light Detection

- Model: YOLOv8n (LISA)
- Dataset: LISA (~10 GB)
- Features:
  - Detects traffic lights in various conditions
  - Classifies state (Red, Yellow, Green, Off)

#### 8. Driver Drowsiness Monitor

- Model: MediaPipe Face Mesh + EAR Algorithm
- Calibration: UTA-RLDD (~23 GB)
- Features:
  - 468-point facial landmark detection
  - Eye Aspect Ratio (EAR) monitoring
  - Fatigue and drowsiness alerts
  - Real-time blink rate analysis

### The ADAS System (Combined Mode)

Run 7 road-based systems simultaneously:
- Lane Detection + LDW
- Forward Collision Warning
- Pedestrian Detection  
- Two-Wheeler Detection
- Animal Awareness
- Traffic Sign Recognition
- Traffic Light Detection

Features:
- Consolidated multi-system visualization
- Unified warning aggregation
- Independent component enable/disable via UI
- Optimized for real-time multi-tasking
 
## How To Run
    
1. Make sure you have Python 3.8+ set up, clone this repository on your local machine, and set up the required datasets.
2. Create a virtual environment, install the required dependencies and run the app:
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Contributing

Contributions are welcome!

## License

Distributed under the MIT License. 
