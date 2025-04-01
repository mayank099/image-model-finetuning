"""
Image processing functionality for the fine-tuning pipeline.
"""

import os
import glob
from typing import Dict, List, Tuple, Optional
from PIL import Image

from .config import ConfigManager


class ImageProcessor:
    """Handles image resizing, padding, and optional text file management."""

    def __init__(self, config: ConfigManager):
        """
        Initialize the image processor.

        Args:
            config: Configuration manager instance
        """
        self.config = config

    def process_images(self, input_dir: Optional[str] = None,
                       output_dir: Optional[str] = None,
                       copy_text_files: bool = True) -> str:
        """
        Process all images in the input directory.

        Args:
            input_dir: Directory containing raw images (optional)
            output_dir: Directory to save processed images (optional)
            copy_text_files: Whether to copy corresponding text files if they exist

        Returns:
            Path to the output directory
        """
        # Use provided directories or defaults from config
        input_dir = input_dir or self.config.get("raw_images_dir")
        output_dir = output_dir or self.config.get("processed_images_dir")

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Get all image files
        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_paths.extend(glob.glob(os.path.join(input_dir, ext)))

        print(f"Found {len(image_paths)} images to process")
        if not image_paths:
            print(f"No images found in {input_dir}")
            return output_dir

        # Process each image
        processed_count = 0
        text_count = 0

        for i, img_path in enumerate(sorted(image_paths), 1):
            try:
                # Get the base filename without extension
                base_filename = os.path.splitext(os.path.basename(img_path))[0]

                # Process the image
                self._resize_and_pad_image(
                    img_path,
                    os.path.join(output_dir, f"img_{i}.png"),
                    self.config.get("output_size"),
                    self.config.get("padding_color")
                )

                processed_count += 1

                # Optionally copy corresponding text file if it exists
                if copy_text_files:
                    txt_path = os.path.join(input_dir, f"{base_filename}.txt")
                    if os.path.exists(txt_path):
                        with open(txt_path, 'r') as src_file:
                            txt_content = src_file.read()

                        # Write to new text file
                        new_txt_path = os.path.join(output_dir, f"img_{i}.txt")
                        with open(new_txt_path, 'w') as dest_file:
                            dest_file.write(txt_content)
                        print(
                            f"Copied {os.path.basename(txt_path)} -> {os.path.basename(new_txt_path)}")
                        text_count += 1

            except Exception as e:
                print(f"Error processing {img_path}: {e}")

        print(f"Processing complete! {processed_count} images converted.")
        if copy_text_files and text_count > 0:
            print(f"Copied {text_count} matching text files.")
        return output_dir

    def _resize_and_pad_image(self, input_path: str, output_path: str,
                              output_size: Tuple[int, int],
                              fill_color: Tuple[int, int, int]) -> None:
        """
        Resize an image while maintaining aspect ratio and add padding.

        Args:
            input_path: Path to input image
            output_path: Path to save processed image
            output_size: Target size as (width, height)
            fill_color: Padding color as RGB tuple
        """
        # Open the image
        img = Image.open(input_path)

        # Get original dimensions
        width, height = img.size

        # Calculate the scaling ratio
        ratio = min(output_size[0] / width, output_size[1] / height)

        # Calculate new size
        new_width = int(width * ratio)
        new_height = int(height * ratio)

        # Resize the image with high-quality resampling
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)

        # Create a new image with padding
        padded_img = Image.new('RGB', output_size, fill_color)

        # Paste the resized image in the center
        paste_x = (output_size[0] - new_width) // 2
        paste_y = (output_size[1] - new_height) // 2
        padded_img.paste(resized_img, (paste_x, paste_y))

        # Save the image
        padded_img.save(output_path, format="PNG")
        print(
            f"Processed {os.path.basename(input_path)} -> {os.path.basename(output_path)}")
