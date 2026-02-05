import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from config.config import Config
from models.yolo_wrapper import YOLOWrapper
from data.prepare_yolo_data import prepare_mtsd_for_yolo

def train_traffic_signs(quick_test=False):

    Config.ensure_dirs()
    
    data_yaml_path = Config.YOLO_SIGNS_DATA / 'data.yaml'
    if data_yaml_path.exists():
        print("Dataset already prepared, skipping data preparation...")
        data_yaml = str(data_yaml_path)
    else:
        print("Preparing MTSD dataset...")
        data_yaml = prepare_mtsd_for_yolo(
            mtsd_root=Config.MTSD_ROOT,
            output_dir=Config.YOLO_SIGNS_DATA
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
        name='yolo_signs'
    )
    
    best_weights = Path('runs/detect/yolo_signs/weights/best.pt')
    if best_weights.exists():
        import shutil
        shutil.copy(best_weights, Config.WEIGHTS_DIR / 'yolo_signs_best.pt')
        print(f"Model saved to {Config.WEIGHTS_DIR / 'yolo_signs_best.pt'}")
    
    print("Training completed.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train traffic sign detection model')
    parser.add_argument('--quick', action='store_true', help='Quick test mode with fewer epochs')
    args = parser.parse_args()
    
    train_traffic_signs(quick_test=args.quick)
