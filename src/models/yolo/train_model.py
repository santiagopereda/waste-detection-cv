from ultralytics import YOLO


def define_model(source='yolov8n.pt'):
    return YOLO(source)


def train_model(model=None, data=None, epochs=100, patience=50,
                batch=16,save_period=10,pretrained=False,
                optimizer='auto',resume=False):

    # Load a pretrained YOLOv8n model
    trained_model = model.train(model=model, data=data, epochs=epochs,
                                    patience=patience, batch=batch,
                                    save_period=save_period,
                                    pretrained=pretrained,
                                    optimizer=optimizer,resume=resume)

    return trained_model


def get_model_classes(model):
    return list(model.names.values())

