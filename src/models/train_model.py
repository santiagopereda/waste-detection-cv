import config as config
import src.params as params
import tarfile
import os

# Import dictionary values specified in config.py file
model_name = config.MODELS_CONFIG[params.MODEL_NAME]['model_name']
pretrained_checkpoint = config.MODELS_CONFIG[params.MODEL_NAME]['pretrained_checkpoint']
base_pipeline_file = config.MODELS_CONFIG[params.MODEL_NAME]['base_pipeline_file']
batch_size = config.MODELS_CONFIG[params.MODEL_NAME]['batch_size']


def download_pretrained_weights():
    """
    Download and extract pretrained model weights.

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
    models_path = os.path.abspath(os.path.join(script_dir, "../../models/research"))
    deploy_path = os.path.join(models_path, 'deploy')
    os.chdir(deploy_path)

    download_tar = os.path.join(params.CHECKPOINT_PATH, pretrained_checkpoint)

    if os.path.isfile(os.path.join(deploy_path, pretrained_checkpoint)):
        print("Checkpoint file already exists")
    else:
        # Download the checkpoint
        os.system(f"wget {download_tar}")

    if os.path.exists(os.path.join(deploy_path, params.CHECKPOINT_FOLDER_NAME)):
        print("Checkpoint folder already exists")
    else:
        # Extract the tar file
        with tarfile.open(pretrained_checkpoint, 'r') as tar:
            tar.extractall()

    print("✅ Pretrained weights downloaded and extracted")

if __name__ == '__main__':
    download_pretrained_weights()
