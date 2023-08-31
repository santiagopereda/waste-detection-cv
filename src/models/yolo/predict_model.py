from ultralytics import YOLO
from PIL import Image
import cv2
import io
import tempfile



def get_predictions(model, source, conf=0.25, iou=0.7, imgsz=640, classes=None, stream=False):
    # Run inference on the source
    results = model(source=source, conf=conf, iou=iou,
                    imgsz=imgsz, classes=classes, stream=False)
    return results


def get_predicted_classes(model, results):
    pred_class_list = []
    for c in results[0].boxes.cls:
        pred_class_list.append(model.names[int(c)])
    return pred_class_list


def get_predicted_shape(results):
    for r in results:
        img_shape = r.orig_shape
    return img_shape


def get_predicted_speed(results):
    for r in results:
        inf_speed = r.speed['inference']
    return inf_speed
        