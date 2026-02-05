from ultralytics import YOLO

class YOLOWrapper:
    
    def __init__(self, model_size='n'):
       
        self.model_size = model_size
        self.model = YOLO(f'yolov8{model_size}.pt')
    
    def train(self, data_yaml, epochs=100, imgsz=640, batch=16, device='mps', name='yolo_model'):
       
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            name=name,
            patience=10,
            save=True,
            plots=True,
            amp=False  
        )
        return results
    
    def load_weights(self, weight_path):
       
        self.model = YOLO(weight_path)
    
    def predict(self, source, conf=0.5, iou=0.45, imgsz=640):
       
        return self.model.predict(source, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
    
    def get_model(self):
        return self.model
