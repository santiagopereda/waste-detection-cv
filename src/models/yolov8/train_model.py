from ultralytics import YOLO
from PIL import Image
from roboflow import Roboflow
from data.data import get_data

API_KEY = 'M5JOZygUKYjMDTlEyr5z'
PROJECT = 'lewagon'
VERSION = 1
DATA_LOCATION = '/home/vuyani/code/Santiago/waste-detection-cv/data/data-train'

# YOLO models -->  yolov8n yolov8s yolov8m yolov8l yolov8x
def define_model(source='yolov8n.yaml'):
    model = YOLO(source)
    return model


def train_model(model,data,epochs=100, patience=50,
                batch=16,save_period=10,pretrained=False,
                optimizer='auto',resume=False):


    # Load a pretrained YOLOv8n model
    trained_model = model.train(model=model, data=data, epochs=epochs,
                                    patience=patience, batch=batch,
                                    save_period=save_period,
                                    pretrained=pretrained,
                                    optimizer=optimizer,resume=resume)

    return trained_model
        # when this function is called it creates a folder with best weights in it
        # as well as the


def get_model_classes(model):
    return list(model.names.values())
