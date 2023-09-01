import src.models.efficientdet.config as config
from src.params_efficientdet import *
import tarfile
import os
from object_detection.utils import label_map_util
import re


# Import dictionary values specified in config.py file
model_name = config.MODELS_CONFIG[MODEL_NAME]['model_name']
pretrained_checkpoint = config.MODELS_CONFIG[MODEL_NAME]['pretrained_checkpoint']
base_pipeline_file = config.MODELS_CONFIG[MODEL_NAME]['base_pipeline_file']
batch_size = config.MODELS_CONFIG[MODEL_NAME]['batch_size']

def download_pretrained_weights_config():
    """
    Download and extract pretrained model weights.
    In addition, download the config file.

    This function downloads the specified pretrained model checkpoint from a URL,
    extracts the checkpoint if it doesn't already exist, and prints a success message.

    Note:
        This script assumes the model configuration is specified in the config file
        and uses parameters from src.params.

    Args:
        None

    Returns:
        None
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_path = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir, os.pardir))
    deploy_path = os.path.join(models_path, 'models', f'{DATA_FOLDER_NAME}', 'models', 'research', 'deploy')

    if not os.path.exists(deploy_path):
        os.makedirs(deploy_path)
    os.chdir(deploy_path)

    download_tar = os.path.join(CHECKPOINT_PATH, pretrained_checkpoint)

    if os.path.isfile(os.path.join(deploy_path, pretrained_checkpoint)):
        print("Checkpoint file already exists")
    else:
        # Download the checkpoint
        os.system(f"wget {download_tar}")

    if os.path.exists(os.path.join(deploy_path, CHECKPOINT_FOLDER_NAME)):
        print("Checkpoint folder already exists")
    else:
        # Extract the tar file
        with tarfile.open(pretrained_checkpoint, 'r') as tar:
            tar.extractall()

    print("✅ Pretrained weights downloaded and extracted")

    #download base training configuration file
    download_config = PATH_CONFIG_FILE + base_pipeline_file
    if os.path.isfile(base_pipeline_file):
        print('Configuration file already exists')
    else:
        os.system(f"wget {download_config}")

    # Change directory to project's parent directory
    parent_directory = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
    os.chdir(parent_directory)

    print("✅ Configuration file downloaded")

def get_num_classes(pbtxt_fname):
    """
    Get the number of classes from a label map file.

    This function loads a label map from the specified .pbtxt file, converts it to categories,
    and then determines the number of unique classes present.

    Args:
        pbtxt_fname (str): File path to the label map in .pbtxt format.

    Returns:
        int: Number of unique classes defined in the label map.
    """
    label_map = label_map_util.load_labelmap(pbtxt_fname)
    categories = label_map_util.convert_label_map_to_categories(
                label_map, max_num_classes=NUMBER_OF_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    return len(category_index.keys())


def write_custom_configuration(train_data,
                               valid_data,
                               labelling_data,
                               batch_size,
                               num_steps
                               ):
    """
    Write a custom configuration file for the pipeline.

    This function reads the contents of the specified pipeline configuration file,
    modifies specific parameters, and writes the modified content to a new file.

    Args:
        batch_size (int, optional): Batch size for training. Defaults to the value specified in params.
        num_steps (int, optional): Number of training steps. Defaults to the value specified in params.

    Returns:
        str: Path to the created custom configuration file.
    """
    print('writing custom configuration file')

    # Get the directory of this file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_path = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir, os.pardir))
    deploy_path = os.path.join(models_path, 'models', f'{DATA_FOLDER_NAME}', 'models', 'research', 'deploy')

    # Construct the path to the pipeline file
    pipeline_fname = os.path.join(deploy_path, base_pipeline_file)

    fine_tune_checkpoint = os.path.join(deploy_path, 'checkpoint', 'ckpt-0')
    num_classes = get_num_classes(labelling_data)

    with open(pipeline_fname) as f:
        s = f.read()

    custom_config_file = os.path.join(deploy_path, 'pipeline_file.config')
    with open(custom_config_file, 'w') as f:
        s = re.sub('fine_tune_checkpoint: ".*?"',
                   'fine_tune_checkpoint: "{}"'.format(fine_tune_checkpoint),
                   s)

        s = re.sub(
            '(input_path: ".*?)(PATH_TO_BE_CONFIGURED/train)(.*?")', 'input_path: "{}"'.format(train_data),
            s)
        s = re.sub(
            '(input_path: ".*?)(PATH_TO_BE_CONFIGURED/val)(.*?")', 'input_path: "{}"'.format(valid_data),
            s)

        s = re.sub(
            'label_map_path: ".*?"', 'label_map_path: "{}"'.format(labelling_data),
            s)

        s = re.sub('batch_size: [0-9]+',
                   'batch_size: {}'.format(batch_size),
                   s)

        s = re.sub('num_steps: [0-9]+',
                   'num_steps: {}'.format(num_steps),
                   s)

        s = re.sub('num_classes: [0-9]+',
                   'num_classes: {}'.format(num_classes),
                   s)

        s = re.sub(
            'fine_tune_checkpoint_type: "classification"',
            'fine_tune_checkpoint_type: "{}"'.format('detection'),
            s)

        f.write(s)

        print("✅ Custom configuration file created")
        pipeline_file = s
        return pipeline_file

# if __name__ == '__main__':
#     download_pretrained_weights_config()
#     write_custom_configuration()
