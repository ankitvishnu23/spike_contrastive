#:!/usr/bin/env python
from setuptools import find_packages, setup

version = None

install_requires = [
    "black",
    "classy-vision",
    "imgaug",
    "matplotlib",
    "opencv-python",
    "pandas",
    "pillow",
    "scikit-image",
    "sklearn",
    "tensorboard_logger",
    "tensorboard",
    "tensorboard_logger",
    "torchtyping",
    "torchvision",
    "typeguard",
]


setup(
    name="contrastive-spikes",
    packages=find_packages(),
    version=version,
    description="contrastive learning for learning spike features",
    author="Ankit Vishnu",
    install_requires=install_requires,  # load_requirements(PATH_ROOT),
    author_email="av3016@columbia.edu",
)
