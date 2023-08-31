.DEFAULT_GOAL := default


#################################################################################
# COMMANDS                                                                      #
#################################################################################
reinstall_package:
	@pip uninstall -y src || :
	@pip install -e .

#################### PACKAGE ACTIONS ###################
# Clone tensorflow github
run_github_clone:
	python -c 'from src.models.clone_tensorflow import clone_tensorflow_models_repo; clone_tensorflow_models_repo()'

# Install tensorflow object detection API
run_obj_det_api:
	python -c 'from src.models.clone_tensorflow import install_object_detection_api; install_object_detection_api()'

# Load dataset from Roboflow
run_load_data:
	python -c "from src.data.make_dataset import load_data; load_data()"
