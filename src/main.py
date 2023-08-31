from ultralytics import YOLO
from src.models.params import *
from src.models.yolo.train_model import get_data, define_model, train_model
from src.models.yolo.predict_model import get_predictions, get_predicted_classes, get_predicted_shape, get_predicted_speed

model_path = '/home/spereda/code/santiagopereda/08-Project/waste-detection-cv/models/yolo_v8'

model_name = ['yolov8n.pt', 'yolov8n-cls.pt', 'yolov8m-cls.pt']

pt_model = model_path +'/'+ model_name[0]

project = '/home/spereda/code/santiagopereda/08-Project/waste-detection-cv/models'

get_data(API_KEY, WORKSPACE, PROJECT, VERSION, LOCATION)

model = YOLO(pt_model)

results = model.train(data='data.yaml', epochs=50, device='cpu', project=project, name=CHECKPOINT_DIR, exist_ok=True, resume=False)