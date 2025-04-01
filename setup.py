"""
Setup script for the image model fine-tuning package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="image_model_finetuning",
    version="0.1.0",
    author="Mayank",
    author_email="mayankyadav0990@gmail.com",
    description="A toolkit for preparing images and fine-tuning image generation models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mayank099/image-model-finetuning",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "Pillow>=9.0.0",
        "openai>=1.0.0",
        "replicate>=0.11.0",
        "python-dotenv>=0.19.0",
    ],
    entry_points={
        "console_scripts": [
            "image-model-finetuning=model_finetuning.cli:main",
        ],
    },
)
