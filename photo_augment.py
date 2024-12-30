import tensorflow as tf
import sys
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import cv2

# Define the augmentation parameters
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode="nearest",
)

# Constants
IMG_WIDTH, IMG_HEIGHT = 30, 30

# Number of augmentations to generate per image
augmentations_per_image = 3

def main():
    """
    Augment all images in input_dir and save them to output_dir.
    """
    # Check command-line arguments
    if len(sys.argv) != 3:
        print("Usage: python augment.py input_dir output_dir")
        sys.exit(1)

    # Assign input and output directories
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    # Augment images
    augment_images(input_dir, output_dir)

def augment_images(input_dir, output_dir):
    # If the output directory doesn't exist, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # List files in the input directory
    img_files = os.listdir(input_dir)
    if len(img_files) == 0:
        print(f"No images found in {input_dir}")
        return

    # Iterate through each image in the input directory
    for img_file in img_files:
        img_path = os.path.join(input_dir, img_file)

        # Load and preprocess the image
        img = load_img(img_path, target_size=(IMG_WIDTH, IMG_HEIGHT)) 
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)  # Reshape to (1, IMG_WIDTH, IMG_HEIGHT, 3)

        # Generate and save augmented images
        i = 0
        for batch in datagen.flow(x, batch_size=1):
            i += 1
            aug_img = batch[0].astype('uint8')
            aug_img_path = os.path.join(output_dir, f'aug_{img_file.split(".")[0]}_{i}.ppm')
            cv2.imwrite(aug_img_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
            if i >= augmentations_per_image:
                break  # Stop after generating the specified number of augmentations

if __name__ == "__main__":
    main()
