"""
Utility functions for image processing.
"""

import os
from typing import Optional
from model_finetuning import ConfigManager, ImageProcessor


def process_images(
    input_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    output_size: tuple = (1024, 1024),
    padding_color: tuple = (0, 0, 0),
    copy_text_files: bool = False,
    config_path: Optional[str] = None
) -> str:
    """
    Process all images in the input directory.

    Args:
        input_dir: Directory containing raw images
        output_dir: Directory to save processed images
        output_size: Target size as (width, height)
        padding_color: Padding color as RGB tuple
        copy_text_files: Whether to copy text files
        config_path: Path to config file

    Returns:
        Path to the output directory
    """
    # Create configuration
    config = ConfigManager(config_path)

    # Override config with function parameters
    if input_dir:
        config.set("raw_images_dir", input_dir)
    if output_dir:
        config.set("processed_images_dir", output_dir)
    if output_size:
        config.set("output_size", output_size)
    if padding_color:
        config.set("padding_color", padding_color)

    # Create directories if they don't exist
    input_dir = config.get("raw_images_dir")
    output_dir = config.get("processed_images_dir")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Initialize image processor
    processor = ImageProcessor(config)

    # Process images
    print(f"Processing images from {input_dir}")
    processed_dir = processor.process_images(
        input_dir=input_dir,
        output_dir=output_dir,
        copy_text_files=copy_text_files
    )

    print(f"Images processed and saved to {processed_dir}")
    return processed_dir


if __name__ == "__main__":
    # This allows the script to be run directly
    import argparse

    parser = argparse.ArgumentParser(
        description="Process images for model training")
    parser.add_argument(
        "--input", help="Input directory containing raw images")
    parser.add_argument(
        "--output", help="Output directory for processed images")
    parser.add_argument("--size", type=int, nargs=2,
                        help="Output size (width height)")
    parser.add_argument("--color", type=int, nargs=3,
                        help="Padding color (R G B)")
    parser.add_argument("--copy-text", action="store_true",
                        help="Copy text files")
    parser.add_argument("--config", help="Path to config file")

    args = parser.parse_args()

    # Convert arguments to parameters
    size = tuple(args.size) if args.size else (1024, 1024)
    color = tuple(args.color) if args.color else (0, 0, 0)

    # Process images
    process_images(
        input_dir=args.input,
        output_dir=args.output,
        output_size=size,
        padding_color=color,
        copy_text_files=args.copy_text,
        config_path=args.config
    )
