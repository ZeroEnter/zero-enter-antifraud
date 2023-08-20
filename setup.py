from distutils.core import setup

from setuptools import find_packages

setup(
    name="zero-enter-antifraud",
    version="0.1.0",
    description="zero-enter-antifraud",
    author="Dmitrii Koriakov",
    author_email="dmitrii.koriakov@uni.lu",
    install_requires=[
        "click==8.1.7",
        "PyYAML",
        "scikit-learn==1.0.2",
        "pandas==1.3.5",
        "numpy==1.21.6",
        "networkx==2.6.3",
        "scipy==1.7.3",
        "torch==1.12.1",
        "matplotlib",
        "jupyter",
        "onnx",
        "ezkl",
    ],
    packages=find_packages(),
)
