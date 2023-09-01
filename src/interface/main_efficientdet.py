from src.data.make_dataset import path_to_data_files
from src.tensorflow_setup import clone_tensorflow_models_repo, install_object_detection_api
from src.models.efficientdet.train_model import download_pretrained_weights_config, write_custom_configuration
from src.params_efficientdet import *
import subprocess

def master_efficientdet():

    train_record_fname, valid_record_fname, test_record_fname, label_map_pbtxt_fname = path_to_data_files()
    print("✅ Data loading is completed")

    clone_tensorflow_models_repo()
    install_object_detection_api()
    print("✅ Tensorflow github repo cloned and API successfully installed.")

    download_pretrained_weights_config()
    print("✅ Pretrained weights is downloaded and checkpoint is saved")

    pipeline_file = write_custom_configuration(train_record_fname,
                                                test_record_fname,
                                                label_map_pbtxt_fname,
                                                BATCH_SIZE,
                                                NUM_STEPS)
    print("✅ Custom configuration file is ready")

if __name__ == '__main__':
    master_efficientdet()
