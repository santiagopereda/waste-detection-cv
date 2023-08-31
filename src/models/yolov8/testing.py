from src.models.yolov8.train_model import define_model,train_model
YAML_PATH = '/home/vuyani/code/Santiago/waste-detection-cv/data.yaml'
from ultralytics import YOLO
if __name__ == '__main__':
    model_1=define_model()
    trained_model = train_model(model=model_1,
                                data=YAML_PATH, epochs=2)
