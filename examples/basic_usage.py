"""
Basic usage example of the image model fine-tuning pipeline.
"""

import os
from dotenv import load_dotenv
from model_finetuning import PipelineManager

# Load environment variables for API keys
load_dotenv()


def main():
    """Run a basic pipeline with default settings."""
    # Initialize the pipeline
    pipeline = PipelineManager()

    # Configure input and output directories
    pipeline.config_manager.set("raw_images_dir", os.path.expanduser(
        "~/Desktop/Image_Model_Training/image-model-finetuning/data/raw_images"))
    pipeline.config_manager.set("processed_images_dir", os.path.expanduser(
        "~/Desktop/Image_Model_Training/image-model-finetuning/data/processed_images"))
    pipeline.config_manager.set("output_dir", os.path.expanduser(
        "~/Desktop/Image_Model_Training/image-model-finetuning/data/output"))

    # Configure subject and model name
    # Set to your subject name or leave blank
    pipeline.config_manager.set("subject_name", "Mayank")
    pipeline.config_manager.set("model_name", "flux-personal-model")
    # Replace with your Replicate username
    pipeline.config_manager.set("replicate_username", "your_username")

    # Run the full pipeline
    print("Starting the image model fine-tuning pipeline...")
    result = pipeline.run_full_pipeline()

    # Print summary of results
    print("\n=== Pipeline Complete ===")
    print(f"Processed images: {result['processed_dir']}")
    print(
        f"Generated descriptions: {len(result['descriptions'])} descriptions")

    if 'prompts' in result and result['prompts']:
        print(
            f"Generated prompts: {len(result['prompts'])} additional prompts")
        print(f"Prompts saved to: {result['prompt_file']}")

    if result['model'] and result['training']:
        print(
            f"Model URL: https://replicate.com/{result['model'].owner}/{result['model'].name}")
        print(f"Training URL: https://replicate.com/p/{result['training'].id}")
    else:
        print("Model training failed or was skipped.")


if __name__ == "__main__":
    main()
