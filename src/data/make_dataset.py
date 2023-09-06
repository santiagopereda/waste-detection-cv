import os
import glob
from datetime import datetime
from roboflow import Roboflow
from src.params_yolo import *
from gcloud import storage, exceptions
from src.models.yolo.utils import get_user_input


def load_data(key=API_KEY,
              workspace=WORKSPACE,
              project=WORKSPACE_PROJECT,
              version=WORKSPACE_PROJECT_VERSION,
              data_type=DATA_TYPE
              ):
    """Load data from Roboflow.

    Args:
        data_type (DATA_TYPE): Type of data to be loaded.
        key (API_KEY): User's API key.
        project (WORKSPACE_PROJECT): Workspace's name at Roboflow.
        version (WORKSPACE_PROJECT_VERSION): Version of the workspace at Roboflow.

    Returns:
        Datasets loaded from Roboflow and stores it in the final_data folder under data folder
    """
    data_dir = os.path.join(HOME, 'data', DATA_FOLDER_NAME)
    print("🔍 Searching for existing data directory")
    if not os.path.exists(data_dir):
        print("data folder doesn't exist, creating directory")
        os.makedirs(data_dir)
        save_location = get_data_folder()
    data_yaml = os.path.join(data_dir, 'data.yaml')
    print(f"🔍 Searching for data.yaml")
    if not os.path.isfile(data_yaml):
        print(f"🌐 Files not found, downloading dataset")
        rf = Roboflow(api_key=key)
        project = rf.workspace(workspace).project(project)
        project.version(version).download(data_type, location=save_location)
        print(f"✅ Data is downloaded and stored in {save_location}")
    else:
        save_location = get_data_folder()
        print(f"✅ Using previously downloaded data")

    return save_location


def get_data_folder():
    """
    Get or create the data folder for the waste-detection-cv project.

    This function constructs the path to the 'data' folder under the waste-detection-cv project.
    If the 'data' folder does not exist, it is created. It then joins this path with `DATA_FOLDER_NAME`
    to get the final data folder path.

    Returns:
        str: The path to the data folder.

    """
    # Go up two levels to waste-detection-cv
    data_dir = os.path.join(HOME, 'data')

    # Check if 'data' directory exists, and create it if not
    if not os.path.exists(data_dir):
        print('📂 Creating a directory "data"')
        os.makedirs(data_dir)

    # Join with DATA_FOLDER_NAME
    data_folder = os.path.join(data_dir, DATA_FOLDER_NAME)

    # Check if 'model' directory exists under data, and create it if not
    if not os.path.exists(data_folder):
        print(f'📂 Creating a directory {DATA_FOLDER_NAME}')
        os.makedirs(data_folder)

    return data_folder


def get_models_folder():
    """
    Get or create the data folder for the waste-detection-cv project.

    This function constructs the path to the 'data' folder under the waste-detection-cv project.
    If the 'data' folder does not exist, it is created. It then joins this path with `DATA_FOLDER_NAME`
    to get the final data folder path.

    Returns:
        str: The path to the data folder.
    """  # Go up two levels to waste-detection-cv
    models_dir = os.path.join(HOME, 'models')

    # Check if 'data' directory exists, and create it if not
    if not os.path.exists(models_dir):
        print('📂 Creating a directory "models"')
        os.makedirs(models_dir)

    # Join with DATA_FOLDER_NAME
    models_folder = os.path.join(models_dir, DATA_FOLDER_NAME)

    # Check if 'model' directory exists under data, and create it if not
    if not os.path.exists(models_folder):
        print(f'📂 Creating a directory {DATA_FOLDER_NAME}')
        os.makedirs(models_folder)

    return models_folder


def path_to_data_files():
    """Use the load_data function to get path of each dataset.

    Returns:
        Path for each of train, test, and validation datasets that were loaded from roboflow
    """
    location = load_data().location
    train_record_fname = location + '/train/waste.tfrecord'
    valid_record_fname = location + '/train/waste.tfrecord'
    test_record_fname = location + '/test/waste.tfrecord'
    label_map_pbtxt_fname = location + '/train/waste_label_map.pbtxt'

    return train_record_fname, valid_record_fname, test_record_fname, label_map_pbtxt_fname


def save_model_gcp():
    try:
        models_location = get_models_folder()
        if DATA_FOLDER_NAME == 'yolov8':
            model_to_save = 'best.pt'
            print(f"🔍 Searching for best weights for model {DATA_FOLDER_NAME}")
            local_directory = os.path.join(
                models_location, CHECKPOINT_DIR, 'weights', model_to_save)
        now = datetime.now().strftime("%d/%m/%Y %H:%M:%S").replace(' ',
                                                                   '-').replace(':', '-').replace('/', '-')
        name = now + '-' + model_to_save
        available_files = glob.glob(os.path.join(HOME, '*.json'))
        if len(available_files) == 1:
            key_location = available_files[0]
        elif len(available_files) > 1:
            key_location = get_user_input(available_files)
        print(f'📡 Establishing connection with Gooogle Cloud Storage')
        client = storage.Client.from_service_account_json(
            json_credentials_path=key_location)
        bucket = client.get_bucket(BUCKET_NAME)
        object_name_in_gcs_bucket = bucket.blob(name)
        object_name_in_gcs_bucket.upload_from_filename(local_directory)
        print(f'💾 {name} saved in {BUCKET_NAME}')
    except UnboundLocalError:
        print(f"🚫 Key not found in {KEY_LOCATION}")
        print("Model not saved")
    except exceptions.NotFound:
        print(
            f"🚫 Can't find {BUCKET_NAME} in buckets, please verify bucket name")
        print("Model not saved")