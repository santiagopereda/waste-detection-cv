<<<<<<< HEAD
.DEFAULT_GOAL := default


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
=======
#################################################################################
# COMMANDS                                                                      #
#################################################################################
## Install Local Packages
reinstall_package:
	@pip uninstall -y src || :
>>>>>>> aee830b28f4ffa72845e58fbda3bf534cb4885ad
