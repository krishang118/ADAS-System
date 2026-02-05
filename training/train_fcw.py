import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from config.config import Config
from models.yolo_wrapper import YOLOWrapper
from data.prepare_yolo_data import prepare_bdd100k_for_yolo

def train_fcw(quick_test=False):

    Config.ensure_dirs()
    
    print("Preparing BDD100K dataset for vehicle detection...")
    data_yaml = prepare_bdd100k_for_yolo(
        bdd_root=Config.BDD100K_ROOT,
        output_dir=Config.YOLO_FCW_DATA,
        vehicle_only=True
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
        name='yolo_fcw'
    )
    
    best_weights = Path('runs/detect/yolo_fcw/weights/best.pt')
    if best_weights.exists():
        import shutil
        shutil.copy(best_weights, Config.WEIGHTS_DIR / 'yolo_fcw_best.pt')
        print(f"Model saved to {Config.WEIGHTS_DIR / 'yolo_fcw_best.pt'}")
    
    print("Training completed.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train FCW vehicle detection model')
    parser.add_argument('--quick', action='store_true', help='Quick test mode with fewer epochs')
    args = parser.parse_args()
    
    train_fcw(quick_test=args.quick)
