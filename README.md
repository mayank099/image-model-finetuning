# Image Model Fine-Tuning Pipeline

A comprehensive Python package for preparing images, analyzing descriptions, generating prompts, and fine-tuning image generation models.

## Overview

This project provides a modular, configurable pipeline for fine-tuning AI image generation models with your own photos. It's specifically designed to work with the FLUX.1 model on Replicate, but the architecture is extensible to support other models.

## Features

- **Image Processing**: Resize and pad images to consistent dimensions while maintaining aspect ratio
- **AI Description Generation**: Automatically generate detailed descriptions for images using OpenAI's vision models
- **Description Analysis**: Analyze image descriptions to identify patterns
- **Prompt Generation**: Generate new, creative prompts for image generation using OpenAI
- **Model Training**: Fine-tune image generation models on Replicate

## Installation

### Option 1: Install from GitHub

```bash
git clone https://github.com/mayank099/image-model-finetuning.git
cd image_model_finetuning
pip install -e .
```

### Option 2: Install Dependencies Directly

```bash
pip install Pillow openai replicate python-dotenv
```

## Configuration

1. Create a `.env` file in your project directory with your API keys:

```
OPENAI_API_KEY=your_openai_api_key
REPLICATE_API_TOKEN=your_replicate_api_token
```

2. (Optional) Create a configuration file (`config.ini`) to customize settings:

```ini
# Directories
raw_images_dir = ~/Desktop/Image_Model_Training/Raw_Images
processed_images_dir = ~/Desktop/Image_Model_Training/Processed_Images
output_dir = ~/Desktop/Image_Model_Training/Output

# Image processing settings
output_size = (1024, 1024)
padding_color = (0, 0, 0)  # Black padding

# Model training settings
replicate_username = your_username
model_name = flux-personal-model
model_visibility = public
model_description = FLUX.1 finetuned on personal photos
training_steps = 1000

# Prompt generation settings
subject_name = Mayank
num_prompts = 5
openai_model = gpt-4o-2024-08-06
```

## Usage

### Command Line Interface

The package provides a command-line interface for easy use:

```bash
# Run the full pipeline with default settings
image-model-finetuning

# Run specific stages of the pipeline
image-model-finetuning --mode process  # Only process images
image-model-finetuning --mode analyze  # Only analyze descriptions
image-model-finetuning --mode generate  # Analyze and generate prompts
image-model-finetuning --mode train    # Only train the model

# Specify custom input/output directories
image-model-finetuning --input /path/to/raw/images --output /path/to/output

# Use a custom configuration file
image-model-finetuning --config my_config.ini

# Set custom training steps
image-model-finetuning --mode train --steps 2000

# Specify subject name
image-model-finetuning --subject "John"

# Set Replicate username and model name
image-model-finetuning --username "your_username" --model-name "my-custom-model"
```

### Directory Structure

The package creates and uses the following directory structure:

```
Raw_Images/                # Your original unprocessed images
├── image1.jpg
├── image2.jpg
└── ...

Processed_Images/          # Created by the pipeline
├── img_1.png              # Resized and padded images
├── img_1.txt              # AI-generated descriptions
├── img_2.png
├── img_2.txt
└── ...

Output/                    # Created by the pipeline
├── data.zip      # ZIP file for training
├── new_generation_prompts.txt  # Additional creative prompts
└── ...
```

## Python API

The package can be used either through its high-level pipeline API or through individual utility functions:

### Pipeline API

```python
from image_model_finetuning import PipelineManager

# Initialize with default or config file settings
pipeline = PipelineManager(config_path="config.ini")

# Run individual stages
processed_dir = pipeline.run_image_processing()
descriptions = pipeline.generate_descriptions()
analysis, all_descriptions = pipeline.run_prompt_analysis()
prompts, output_file = pipeline.run_prompt_generation()
model, training = pipeline.run_model_training()

# Or run the full pipeline
result = pipeline.run_full_pipeline()
```

## Workflow

This package supports a 3-step workflow for creating custom image generation models:

1. **Image Processing**: Raw images are resized, padded, and prepared for training.

2. **Description Generation**: OpenAI's vision models analyze each image and create detailed descriptions.

3. **Model Training**: Processed images and their descriptions are used to fine-tune an image generation model on Replicate.

The full workflow looks like this:

```
Raw Images → Process Images → Generate Descriptions → Train Model → Custom AI Model (via Replicate)
```

You can also add optional steps like prompt analysis and creative prompt generation to enhance your training data.

## Project Structure

The code is organized into a modular package structure:

```
image-model-finetuning/
├── data/                     # Data directories (Excluded from Git)
│   ├── output/               # Output files and training data
│   ├── processed_images/     # Processed images with descriptions
│   └── raw_images/           # Input images
│
├── examples/                 # Example scripts
│   ├── init.py               # Package initialization for examples
│   ├── basic_usage.py        # Basic pipeline example
│   └── custom_workflow.py    # Custom workflow example
│
├── model_finetuning/         # Main package directory
│   ├── pycache/              # Python cache files
│   ├── templates/            # Template files
│   │   ├── init.py           # Package initialization for templates
│   │   ├── description_template.txt  # Template for image descriptions
│   │   ├── system_prompt_template.txt  # Template for system prompts
│   │   └── user_prompt_template.txt    # Template for user prompts
│   │
│   ├── init.py               # Package initialization
│   ├── cli.py                # Command-line interface
│   ├── config.py             # Configuration management
│   ├── image_processor.py    # Image processing functionality
│   ├── model_trainer.py      # Model training on Replicate
│   ├── pipeline.py           # Pipeline orchestration
│   └── prompt_analyzer.py    # Description generation and analysis
│
├── utils/                    # Utility scripts and functions
│   ├── pycache/              # Python cache files
│   ├── init.py               # Utility module initialization
│   ├── analyze_prompts.py    # Prompt analysis utility
│   ├── generate_descriptions.py  # Description generation utility
│   ├── process_images.py     # Image processing utility
│   └── train_model.py        # Model training utility
│
├── config.ini                # Configuration settings
├── README.md                 # Project documentation
├── requirements.txt          # Package dependencies
└── setup.py                  # Package installation script
```

The modular design allows you to use the package as a whole or import specific components for custom workflows.

## Requirements

- Python 3.7+
- Pillow (for image processing)
- OpenAI Python SDK (for prompt generation)
- Replicate Python SDK (for model training)
- python-dotenv (for environment variables)

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.