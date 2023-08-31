import os
import pathlib
import subprocess
import platform
import src.params as params

##################  clone tensorflow github repository of models  ##################
def clone_tensorflow_models_repo(url=params.REPO_URL):
    """
    This function is used to clone the git repository of tensorflow
    models. The function checks if the 'models' folder exists or not.
    If the 'models' folder does not exist then cloning is happening.
    @Keyword argument = url: url is the ssh key of the tensorflow repo.
    """
    if "models" in pathlib.Path.cwd().parts:
        while "models" in pathlib.Path.cwd().parts:
            os.chdir('..')
    elif not pathlib.Path('models').exists():
        subprocess.run(['git', 'clone', '--depth', '1', url])

    #Check if the 'models' folder indeed cloned or not
    if pathlib.Path("models").exists():
        print("Repository is cloned.")
    else:
        print("Repository is not cloned.")

    print("✅ Tensorflow's model repository cloned")


def install_object_detection_api():
    # Calculate the path to the models directory (two steps up from the current script location)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_path = os.path.abspath(os.path.join(script_dir, "../../models/research"))
    protos_path = os.path.join(models_path, "object_detection/protos")

    os.chdir(models_path)

    subprocess.run(f"protoc {protos_path}/*.proto --python_out=.", shell=True)
    subprocess.run(f"cp {models_path}/object_detection/packages/tf2/setup.py .", shell=True)
    subprocess.run("python -m pip install .", shell=True)

    print("✅ Object detection API installed")

    #If machine is M1 print warning
    if platform.machine() == 'arm64':
        print("🚫 Seperate process is needed to install tensorflow-io for M1.")
        # Clone tensorflow/io and install it
        # subprocess.run("git clone https://github.com/tensorflow/io", shell=True)
        # os.chdir("io")
        # subprocess.run("python setup.py build", shell=True)
        # subprocess.run("python setup.py install", shell=True)
        # print("✅ tensorflow/io installed for Mac M1")


if __name__ == "__main__":
    clone_tensorflow_models_repo(params.REPO_URL)
    install_object_detection_api()
