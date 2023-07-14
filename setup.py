#:!/usr/bin/env python
from setuptools import find_packages, setup

version = None

install_requires = [
    "black",
    "classy-vision",
    "fiftyone",
    "h5py",
    "hydra-core",
    "imgaug",
    "kornia",
    "matplotlib",
    "moviepy",
    "opencv-python",
    "pandas",
    "pillow",
    "lightning",
    "scikit-image",
    "sklearn",
    "tensorboard_logger",
    "tensorboard",
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
