"""
Setup script for FreqPress
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="freqpress",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="FreqPress: Adversarial Defense via Frequency-Domain Preprocessing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/freqpress",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.10.0",
        "torchvision>=0.11.0",
        "numpy>=1.21.0",
        "Pillow>=9.0.0",
        "timm>=0.6.0",
        "torchattacks>=3.4.0",
        "matplotlib>=3.5.0",
    ],
)
