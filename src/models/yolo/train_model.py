import yaml
import glob
from ultralytics import YOLO
from src.params_yolo import *
from src.models.yolo.utils import get_user_input
from src.data.make_dataset import get_models_folder

def define_model(model_target = None, name=None) -> YOLO:
    """
    Defines and initializes a YOLO model using the provided source.
    Args:
        model_target (str): The source for initializing the YOLO model.
    Returns:
        YOLO: The initialized YOLO model.
    """    
    models_location = get_models_folder()
    try:
        model_dir = os.path.join(models_location, CHECKPOINT_DIR, 'weights')
        if model_target == 'best' and name == None:
            source = os.path.join(model_dir, 'best.pt')
            model = YOLO(source)
            # Print a message indicating the model being used
            print(f"✅ Initiatizing last trained models with the best weights")
            return model
        elif model_target == 'local' and name == None:
            available_files = glob.glob(os.path.join(models_location,'*.pt'))
            if len(available_files) == 1:
                model = YOLO(available_files[0])
            elif len(available_files) > 1:
                selected_file = get_user_input(available_files)
                model = YOLO(selected_file)
            return model
        elif model_target == None and name != None:
            model = YOLO(name)
            (f"✅ Initializing pretrained model: {name}")
            return model
        else:
            model = YOLO('yolov8n.pt')
            (f"✅ Initializing pretrained model: 'yolov8n.pt'")
            return model
            
    except TypeError:
        model = None
        print(f"🚫 Please specify the model name")
        
    except FileNotFoundError:
        model = None
        print(f"🚫 Please verify source path: {model_dir}")
        
    except UnboundLocalError:
        print(f"🚫 Select model with pretrained weights to initialize")
        model = None
        

def get_dataset_classes():
    """
    Get the number of classes from a label map file.

    This function loads a label map from the specified .pbtxt file, converts it to categories,
    and then determines the number of unique classes present.

    Args:
        pbtxt_fname (str): File path to the label map in .pbtxt format.

    Returns:
        int: Number of unique classes defined in the label map.
    """

    data_yaml = os.path.join(HOME, 'data', DATA_FOLDER_NAME, 'data.yaml')
    
    with open(data_yaml, 'r') as file:
        data = yaml.safe_load(file)
        
    dataset_classes = data['names']
    
    
    return dataset_classes




def train_model(model=None, data=None, epochs=1000, patience=10,
                batch=10, imgsz=640, save_period=1, device=None, project=None, name=None,
                exist_ok=True, pretrained=False, optimizer='auto', resume=False,
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
        if data == None:
            data_dir = os.path.join(HOME, 'data', DATA_FOLDER_NAME)
            data = os.path.join(data_dir, 'data.yaml')            
        if project == None:
            project = os.path.join(HOME, 'models', 'yolov8')
        if name == None:
            name=CHECKPOINT_DIR
        trained_model = model.train(data=data, epochs=epochs, patience=patience,
                                                batch=batch, imgsz=imgsz, save_period=save_period, device=device,
                                                project=project, name=name, exist_ok=exist_ok, pretrained=pretrained,
                                                optimizer=optimizer, resume=resume, lr0=lr0, lrf=lrf, dropout=dropout)
    except (RuntimeError):
        print(f"🚫 'data.yaml' file not found\n", "source path: {data}")
        trained_model = None

    except AttributeError:
        print(f"🚫 Can't train without and initialized model")
        trained_model = None
        
    return trained_model