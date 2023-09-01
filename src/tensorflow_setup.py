import os
import pathlib
import subprocess
import platform
from src.params_efficientdet import *


##################  clone tensorflow github repository of models  ##################
def clone_tensorflow_models_repo(url=REPO_URL):
    """
    Clone the TensorFlow Models Repository.

    This function clones the TensorFlow Models repository from the specified URL.
    It checks if the 'models' folder exists in the current directory. If the folder
    does not exist, it creates it and clones the repository there.

    Args:
        url (str, optional): The URL of the TensorFlow Models repository to clone.

    Returns:
        None
    """
    models_folder = os.path.join(os.getcwd(), 'models', DATA_FOLDER_NAME)

    if os.path.exists(models_folder):
        print("Repository already exists.")
    else:
        print("Repository has not been cloned yet. Cloning starts now")
        os.makedirs(models_folder)
        os.chdir(models_folder)
        subprocess.run(['git', 'clone', '--depth', '1', url])

    print("✅ Tensorflow's model repository successfully cloned")


def install_object_detection_api():
    """
    Install the Object Detection API by compiling protocol buffer files and running setup.py.

    This function calculates the path to the models directory, compiles the protocol buffer
    files in the object_detection/protos subdirectory, copies the setup.py file from the
    object_detection/packages/tf2 directory, and installs the Object Detection API using pip.

    It also provides a warning message for users with ARM64 (M1) machines, indicating that
    a separate process may be needed to install tensorflow-io.

    Returns:
        None
    """
    # Calculate the path to the models directory (two steps up from the current script location)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_path = os.path.abspath(os.path.join(script_dir, f"../models/{DATA_FOLDER_NAME}/models/research"))
    protos_path = os.path.join(models_path, "object_detection/protos")
    os.chdir(models_path)
    subprocess.run(f"protoc --python_out=. --proto_path={protos_path} {protos_path}/*.proto", shell=True)
    subprocess.run(f"cp {models_path}/object_detection/packages/tf2/setup.py .", shell=True)
    subprocess.run("python -m pip install .", shell=True)

    print("✅ Object detection API installed")

    #If machine is M1 print warning
    if platform.machine() == 'arm64':
        print("🚫 Seperate process is needed to install tensorflow-io for M1.")
    #     # Clone tensorflow/io and install it
    #     # subprocess.run("git clone https://github.com/tensorflow/io", shell=True)
    #     # os.chdir("io")
    #     # subprocess.run("python setup.py build", shell=True)
    #     # subprocess.run("python setup.py install", shell=True)
    #     # print("✅ tensorflow/io installed for Mac M1")

    # Change directory to project's parent directory
    parent_directory = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
    os.chdir(parent_directory)


# if __name__ == "__main__":
#     clone_tensorflow_models_repo(REPO_URL)
#     install_object_detection_api()
