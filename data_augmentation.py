import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import cv2

# Constants
IMG_WIDTH, IMG_HEIGHT = 30, 30
NUM_CATEGORIES = 43
TARGET_PER_CATEGORY = 3000 

# Create an ImageDataGenerator for augmentation without horizontal flip
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode='nearest'
)

def main():
    """
    Augment all images in input_dir and save in output_dir
    """
    # check usage
    if len(sys.argv) != 3:
        print("Usage: python augment.py input_dir output_dir")
        sys.exit(1)

    # assign input and output directories
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    # augment images
    augment_images(input_dir, output_dir)
    
def augment_images(input_dir, output_dir):
    """
    Perform data augmentation on images in the input directory and save augmented images to the output directory.

    Args:
        input_dir (str): Path to the input directory containing subdirectories for each category.
        output_dir (str): Path to the output directory where augmented images will be saved.
    """
    # for each category
    for category in range(NUM_CATEGORIES):
        category_dir = os.path.join(input_dir, str(category))
        output_category_dir = os.path.join(output_dir, str(category))

        # if the folder doesn't exist create it
        if not os.path.exists(output_category_dir):
            os.makedirs(output_category_dir)

        # calculate the number of augmentations_per_image
        category_size = len(os.listdir(category_dir))
        if category_size == 0:
            print(f"No images found in category {category}")
            continue

        augmentations_per_image = max(1, int(TARGET_PER_CATEGORY / category_size))

        # for each image
        for img_file in os.listdir(category_dir):

            # load image
            img_path = os.path.join(category_dir, img_file)
            img = load_img(img_path)
            
            # reshape image
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)  # Reshape to (1, IMG_WIDTH, IMG_HEIGHT, 3)

            # Generate augmented images
            i = 0
            for batch in datagen.flow(x, batch_size=1):
                i += 1
                aug_img = batch[0].astype('uint8')
                aug_img_path = os.path.join(output_category_dir, f'aug_{img_file.split(".")[0]}_{i}.ppm')
                cv2.imwrite(aug_img_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
                if i >= augmentations_per_image:
                    break  # Stop after generating the specified number of augmentations

if __name__ == "__main__":
    main()
