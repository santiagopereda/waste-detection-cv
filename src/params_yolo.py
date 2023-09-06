import os


##################  CONSTANTS  ##################
HOME = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))


##################  ROBOFLOW  ##################
API_KEY = os.environ.get("API_KEY")
WORKSPACE = os.environ.get("WORKSPACE")
WORKSPACE_PROJECT = os.environ.get("WORKSPACE_PROJECT")
WORKSPACE_PROJECT_VERSION = os.environ.get("WORKSPACE_PROJECT_VERSION")
DATA_TYPE = os.environ.get("DATA_TYPE")
DATA_FOLDER_NAME = os.environ.get("DATA_FOLDER_NAME")


##################  YOLOV8  ##################
LOCATION = os.environ.get("LOCATION")
CHECKPOINT_DIR =os.environ.get("CHECKPOINT_DIR")
MODEL_TARGET = os.environ.get("MODEL_TARGET")
MODEL_SAVE = os.environ.get("MODEL_SAVE")
EPOCHS = os.environ.get("EPOCHS")
PATIENCE = os.environ.get("PATIENCE")
DEVICE = os.environ.get("DEVICE")
OPTIMIZER = os.environ.get("OPTIMIZER")


##################  GCP  ##################
if os.environ.get("KEY_LOCATION") == None:
    KEY_LOCATION = os.path.dirname(os.path.realpath(__file__))
else:
    KEY_LOCATION = os.environ.get("KEY_LOCATION")
BUCKET_NAME = os.environ.get("BUCKET_NAME")

################## API on gcloud #################
GCR_IMAGE = os.environ.get("GCR_IMAGE")
GCR_REGION = os.environ.get("GCR_REGION")
GCR_MEMORY = os.environ.get("GCR_MEMORY")
SERVICE_URL = os.environ.get("SERVICE_URL")
