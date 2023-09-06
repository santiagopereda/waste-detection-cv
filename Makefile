#################################################################################
# COMMANDS                                                                      #
#################################################################################
reinstall_package:
	@pip uninstall -y src || :
	@pip install -e .

#################### PACKAGE ACTIONS ###################
# Run main_efficientdet.py file
run_main_yolo:
	python -c 'from src.main import master_yolo; master_yolo()'
