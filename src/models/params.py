import os
import numpy as np

##################  VARIABLES  ##################
API_KEY = os.environ.get("API_KEY")
WORKSPACE = os.environ.get("WORKSPACE")
PROJECT = os.environ.get("PROJECT")
VERSION = os.environ.get("VERSION")
LOCATION = os.environ.get("LOCATION")
CHECKPOINT_DIR =os.environ.get("CHECKPOINT_DIR")

##################  CONSTANTS  #####################




################## VALIDATIONS #################

def validate_env_value(env, valid_options):
    env_value = os.environ[env]
    if env_value not in valid_options:
        raise NameError(f"Invalid value for {env} in `.env` file: {env_value} must be in {valid_options}")