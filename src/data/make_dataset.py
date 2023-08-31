import os
from roboflow import Roboflow
import src.params as params


def load_data(key=params.API_KEY,
              project=params.WORKSPACE_PROJECT,
              version=params.WORKSPACE_PROJECT_VERSION,
              data_type=params.DATA_TYPE
              ):
    """Load data from Roboflow.

    Args:
        data_type (params.DATA_TYPE): Type of data to be loaded.
        key (params.API_KEY): User's API key.
        project (params.WORKSPACE_PROJECT): Workspace's name at Roboflow.
        version (params.WORKSPACE_PROJECT_VERSION): Version of the workspace at Roboflow.

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
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(current_dir, params.DATA_FOLDER_NAME)

    return data_folder

def path_files():
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
#     loaded_dataset = load_data(params.API_KEY,
#                                params.WORKSPACE_PROJECT,
#                                params.WORKSPACE_PROJECT_VERSION,
#                                params.DATA_TYPE)


