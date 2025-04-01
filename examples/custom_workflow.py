"""
Example of a custom workflow using the image model fine-tuning utilities.
"""

import os
from dotenv import load_dotenv
from utils import (
    process_images,
    generate_descriptions,
    analyze_prompts,
    generate_new_prompts,
    train_model
)

# Load environment variables for API keys
load_dotenv()


def custom_workflow():
    """
    Run a custom workflow using the individual utility functions.
    This demonstrates how to build a custom pipeline for specific needs.
    """
    # Define directories
    raw_dir = os.path.expanduser(
        "~/Desktop/Image_Model_Training/image-model-finetuning/data/raw_images")
    processed_dir = os.path.expanduser(
        "~/Desktop/Image_Model_Training/image-model-finetuning/data/processed_images")
    output_dir = os.path.expanduser(
        "~/Desktop/Image_Model_Training/image-model-finetuning/data/output")

    # Step 1: Process Images (resize, padding)
    print("\n===== Step 1: Processing Images =====")
    processed_dir = process_images(
        input_dir=raw_dir,
        output_dir=processed_dir,
        output_size=(1024, 1024),
        padding_color=(0, 0, 0),
        copy_text_files=False  # Don't copy existing text files
    )

    # Step 2: Generate Descriptions for the processed images
    print("\n===== Step 2: Generating Descriptions =====")
    # Custom template for a specific subject, style, or content
    template = """
    Please create a detailed, single-sentence description of this image.
    Focus on capturing the following aspects:
    - The main subject and their actions/pose
    - The environment or setting
    - The lighting and atmosphere
    - Any unique visual elements
    
    Create a description that would help an AI model generate a similar image.
    """

    descriptions = generate_descriptions(
        image_dir=processed_dir,
        subject_name="Mayank",  # Change to your subject or remove
        template_file=None,  # Using the string template above instead of a file
        overwrite=True  # Overwrite any existing descriptions
    )

    # Step 3: Analyze the generated descriptions
    print("\n===== Step 3: Analyzing Descriptions =====")
    analysis, all_descriptions = analyze_prompts(
        descriptions_dir=processed_dir,
        subject_name="Mayank"  # Change to your subject or remove
    )

    # Step 4: Generate new, more creative prompts based on the analysis
    print("\n===== Step 4: Generating Creative Prompts =====")
    new_prompts, prompts_file = generate_new_prompts(
        analysis=analysis,
        descriptions=all_descriptions,
        output_file=os.path.join(output_dir, "creative_prompts.txt"),
        num_prompts=10  # Generate 10 new creative prompts
    )

    # Step 5: Train the model using the processed images and descriptions
    print("\n===== Step 5: Training the Model =====")
    # Only proceed if we have the Replicate API token
    if os.environ.get("REPLICATE_API_TOKEN"):
        model, training = train_model(
            images_dir=processed_dir,
            replicate_username="your_username",  # Replace with your username
            model_name="custom-image-model",
            model_description="A custom fine-tuned image generation model",
            training_steps=1500  # Increase steps for better results
        )
    else:
        print("Skipping model training: REPLICATE_API_TOKEN not set")

    print("\n===== Custom Workflow Complete =====")


if __name__ == "__main__":
    custom_workflow()
