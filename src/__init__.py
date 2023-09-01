from os.path import isfile
<<<<<<< HEAD
from os.path import dirname

version_file = '{}/version.txt'.format(dirname(__file__))

if isfile(version_file):
    with open(version_file) as version_file:
        __version__ = version_file.read().strip()
=======
from os.path import dirname
>>>>>>> aee830b28f4ffa72845e58fbda3bf534cb4885ad
