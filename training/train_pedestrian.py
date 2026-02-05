import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from config.config import Config
from models.yolo_wrapper import YOLOWrapper
from data.prepare_yolo_data import prepare_caltech_for_yolo

def train_pedestrian(quick_test=False):
    Config.ensure_dirs()
    
    print("Preparing Caltech Pedestrian dataset...")
    data_yaml = prepare_caltech_for_yolo(
        caltech_root=Config.CALTECH_ROOT,
        output_dir=Config.YOLO_PED_DATA
    )
    
    print("Initializing YOLOv8n...")
    yolo = YOLOWrapper(model_size='n')
    
    epochs = Config.YOLO_EPOCHS_QUICK if quick_test else Config.YOLO_EPOCHS
    print(f"Training for {epochs} epochs on {Config.DEVICE}...")
    
    results = yolo.train(
        data_yaml=data_yaml,
        epochs=epochs,
        imgsz=Config.YOLO_IMG_SIZE,
        batch=Config.BATCH_SIZE,
        device=Config.DEVICE,
        name='yolo_pedestrian'
    )
    
    best_weights = Path('runs/detect/yolo_pedestrian/weights/best.pt')
    if best_weights.exists():
        import shutil
        shutil.copy(best_weights, Config.WEIGHTS_DIR / 'yolo_pedestrian_best.pt')
        print(f"Model saved to {Config.WEIGHTS_DIR / 'yolo_pedestrian_best.pt'}")
    
    print("Training completed.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train pedestrian detection model')
    parser.add_argument('--quick', action='store_true', help='Quick test mode with fewer epochs')
    args = parser.parse_args()
    
    train_pedestrian(quick_test=args.quick)
