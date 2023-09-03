#################################################################################
# COMMANDS                                                                      #
#################################################################################
reinstall_package:
	@pip uninstall -y src || :
	@pip install -e .


#################### PACKAGE ACTIONS ###################
# Run main_efficientdet.py file
run_main_efficientdet:
	python -c 'from src.interface.main_efficientdet import master_efficientdet; master_efficientdet()'

