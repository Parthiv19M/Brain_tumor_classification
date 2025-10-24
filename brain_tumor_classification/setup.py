from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="brain_tumor_classification",
    version="0.1.0",
    author="Parthiv",
    author_email="your.email@example.com",
    description="A deep learning model for brain tumor classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Parthiv19M/Brain_tumor_classification",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "tensorflow>=2.8.0",
        "numpy>=1.19.5",
        "pandas>=1.3.5",
        "matplotlib>=3.5.1",
        "scikit-learn>=1.0.2",
        "opencv-python>=4.5.5.64",
        "jupyter>=1.0.0",
        "seaborn>=0.11.2",
        "pillow>=8.4.0",
    ],
)
