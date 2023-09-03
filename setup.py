from setuptools import find_packages, setup
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(
    name='src',
    version='0.1',
    description='Deep learning-based waste detection in natural and urban environments',
    license="MIT",
    author="Le Wagon",
    author_email="contact@lewagon.org",
    install_requires=requirements,
    packages=find_packages())