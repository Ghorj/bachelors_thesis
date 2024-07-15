import os
import sys
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from preprocess import histogram_equalization, clahe, gamma_correction, log_transform, adaptive_histogram_equalization

def preprocess_image(img, method):
    """
    Preprocess images
    """
    # methods
    if method == "histogram_equalization":
        return histogram_equalization(img)
    elif method == "clahe":
        return clahe(img)
    elif method == "gamma":
        return gamma_correction(img, gamma=2.0)  # Adjust gamma value as needed
    elif method == "log":
        return log_transform(img)
    elif method == "adaptive_histogram_equalization":
        return adaptive_histogram_equalization(img)
    else:
        raise ValueError(f"Unknown preprocessing method: {method}")

def main():
    """
    Take random photos from a directory and process them with every preprocessing method in preprocess.py
    """
    # check usage
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python preprocessed_samples.py input_directory num_photos [output_file]")
        sys.exit(1)

    # assign variables
    input_dir = sys.argv[1]
    num_photos = int(sys.argv[2])
    output_file = sys.argv[3] if len(sys.argv) == 4 else "preprocessed_photos.png"

    # Collect all image paths
    image_paths = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.ppm')):
                image_paths.append(os.path.join(root, file))

    if len(image_paths) < num_photos:
        print(f"Number of photos requested ({num_photos}) exceeds available photos ({len(image_paths)}).")
        sys.exit(1)

    # Select random images
    selected_paths = random.sample(image_paths, num_photos)

    # method names
    methods = ["histogram_equalization", "clahe", "gamma", "log", "adaptive_histogram_equalization"]
    
    # 
    fig, axes = plt.subplots(num_photos, len(methods) + 1, figsize=(20, 5 * num_photos))

    # for each image path
    for i, img_path in enumerate(selected_paths):

        # load image and convert to uint8 for OpenCV functions
        img = load_img(img_path)
        img_array = img_to_array(img).astype('uint8')
        axes[i, 0].imshow(img_array.astype('uint8'))
        axes[i, 0].set_title("Original")
        axes[i, 0].axis("off")
        
        # for each method
        for j, method in enumerate(methods):

            # apply method
            preprocessed_img = preprocess_image(img_array, method)

            # add image to the grid
            axes[i, j + 1].imshow(preprocessed_img.astype('uint8'))
            axes[i, j + 1].set_title(method)
            axes[i, j + 1].axis("off")

    # save grid to output file
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Preprocessed photos saved to {output_file}")

if __name__ == "__main__":
    main()
