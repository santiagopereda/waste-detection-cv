from ultralytics import YOLO
from roboflow import Roboflow
from src.models.params import *


def get_data(api_key: str, workspace: str, project: str, version: str,data_type: str, location: str) -> str:
    """
    Downloads a dataset version from Roboflow using the provided API key and saves it to the specified location.

    Args:
        api_key (str): Your Roboflow API key for authentication.
        workspace (str): The name of the workspace containing the project.
        project (str): The name of the project containing the desired dataset version.
        version (str): The version of the dataset to be downloaded.
        location (str): The local directory where the dataset will be saved.

    Returns:
        str: Confirmation message indicating the location where the data was saved.
    """
    # Initialize the Roboflow instance with the provided API key
    rf = Roboflow(api_key=api_key)

    # Access the specified workspace and project using the Roboflow instance
    project = rf.workspace(workspace).project(project)

    # Access the desired dataset version within the project and download it to the specified location
    dataset = project.version(version).download(data_type, location)

    # Print a success message
    print(f"✅ Data saved in {location}")

    # Return a confirmation message
    return dataset


def define_model(source: str) -> YOLO:
    """
    Defines and initializes a YOLO model using the provided source.
    Args:
        source (str): The source for initializing the YOLO model.
    Returns:
        YOLO: The initialized YOLO model.
    """
    # Initialize the YOLO model using the provided source
    model = YOLO(source)

    # Print a message indicating the model being used
    print(f"✅ Now training on model: {source.split('/')[-1]}")

    # Return the initialized model
    return model


def train_model(model=None, data=None, epochs=100, patience=10,
                batch=16, imgsz=640, save_period=1, device=None, project=None, name=None,
                exist_ok=False, pretrained=False, optimizer='auto', resume=False,
                lr0=0.01, lrf=0.01, dropout=0.0):
    """
    Trains a model using the provided data and settings.

    Args:
        model: The model to be trained. (Assuming it has a train method)
        data: The training data. (Assuming it's in a format suitable for the model's train method)
        epochs (int): Number of training epochs.
        patience (int): Number of epochs with no improvement after which training will be stopped.
        batch (int): Batch size for training.
        imgsz (int): Size of the input images.
        save_period (int): Number of epochs between saving checkpoints.
        device: Device to use for training.
        project: The project to which the trained model will be saved.
        name: Name of the trained model.
        exist_ok (bool): If True, overwrite existing model checkpoints.
        pretrained (bool): If True, use pretrained weights if available.
        optimizer (str): The optimizer to use.
        resume (bool): If True, resume training from a checkpoint.
        lr0 (float): Initial learning rate.
        lrf (float): Final learning rate as a fraction of the initial rate.
        dropout (float): Dropout rate.

    Returns:
        The trained model.
    """
    try:
        trained_model = model.train(data=data, epochs=epochs, patience=patience,
                                    batch=batch, imgsz=imgsz, save_period=save_period, device=device,
                                    project=project, name=name, exist_ok=exist_ok, pretrained=pretrained,
                                    optimizer=optimizer, resume=resume, lr0=lr0, lrf=lrf, dropout=dropout)

    except (FileNotFoundError, RuntimeError):
        print("Add Model or Data to train on")
        traceback.print_exc()
        trained_model = None

    except Exception:
        traceback.print_exc()
        trained_model = None
        
    return trained_model