import os
from roboflow import Roboflow
from src.params_efficientdet import *


def load_data(key=API_KEY,
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
    save_location = get_data_folder()
    rf = Roboflow(api_key=key)
    project = rf.workspace().project(project)
    dataset = project.version(version).download(data_type, location=save_location)

    print("✅ Data is loaded and stored in the folder named final folder")

    return dataset


def get_data_folder():
    """
    Get or create the data folder for the waste-detection-cv project.

    This function constructs the path to the 'data' folder under the waste-detection-cv project.
    If the 'data' folder does not exist, it is created. It then joins this path with `DATA_FOLDER_NAME`
    to get the final data folder path.

    Returns:
        str: The path to the data folder.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))  # Go up two levels to waste-detection-cv
    data_dir = os.path.join(parent_dir, 'data')

    # Check if 'data' directory exists, and create it if not
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    data_folder = os.path.join(data_dir, DATA_FOLDER_NAME)  # Join with DATA_FOLDER_NAME

    # Check if 'efficientdet' directory exists under data, and create it if not
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    return data_folder


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

# if __name__ == '__main__':
#     loaded_dataset = load_data(API_KEY,
#                                WORKSPACE_PROJECT,
#                                WORKSPACE_PROJECT_VERSION,
#                                DATA_TYPE)
