import os

##################  Tensorflow GIT REPO  ##################
REPO_URL = os.environ.get("REPO_URL")

##################  Roboflow  #############################
API_KEY = os.environ.get("API_KEY")
WORKSPACE_PROJECT = os.environ.get("WORKSPACE_PROJECT")
WORKSPACE_PROJECT_VERSION = os.environ.get("WORKSPACE_PROJECT_VERSION")
DATA_TYPE = os.environ.get("DATA_TYPE")
DATA_FOLDER_NAME = "final_data"

##################  Specify training configurations  #############################
MODEL_NAME = os.environ.get("chosen_model")
NUM_STEPS = os.environ.get("num_steps")
NUM_EVAL_STEPS = os.environ.get("num_eval_steps")
BATCH_SIZE = os.environ.get("batch_size")
CHECKPOINT_PATH = os.environ.get("pre_trained_checkpoint")
CHECKPOINT_FILE_NAME = os.environ.get('checkpoint_file')
CHECKPOINT_FOLDER_NAME = os.environ.get('checkpoint_folder')
