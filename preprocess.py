import os
import sys
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img

# Preprocessing methods
def histogram_equalization(img):
    img_hsi = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    H, S, I = cv2.split(img_hsi)
    eq_I = cv2.equalizeHist(I)
    eq_img = cv2.merge([H, S, eq_I])
    return cv2.cvtColor(eq_img, cv2.COLOR_HSV2RGB)

def clahe(img):
    img_hsi = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(4, 4))
    H, S, I = cv2.split(img_hsi)
    eq_I = clahe.apply(I)
    eq_img = cv2.cvtColor(cv2.merge([H, S, eq_I]), cv2.COLOR_HSV2RGB)
    return eq_img

def gamma_correction(img, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)

def log_transform(img):
    c = 255 / np.log(1 + np.max(img))
    log_image = c * (np.log(img + 1))
    return np.array(log_image, dtype=np.uint8)

def adaptive_histogram_equalization(img):
    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img_lab[:, :, 0] = clahe.apply(img_lab[:, :, 0])
    return cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)

def conversion(img):
    # modify in main
    return img

def preprocess_image(img, method):
    if isinstance(img, np.ndarray):  # Check if it's already a NumPy array
        pass
    else:
        img = np.array(img)  # Convert PIL image to NumPy array
        
    if method == "he":
        return histogram_equalization(img)
    elif method == "clahe":
        return clahe(img)
    elif method == "gamma":
        return gamma_correction(img, gamma=3.0)  # Adjust gamma value as needed
    elif method == "log":
        return log_transform(img)
    elif method == "ahe":
        return adaptive_histogram_equalization(img)
    elif method == "conversion":
        return conversion(img)
    else:
        raise ValueError(f"Unknown preprocessing method: {method}. \nAvailable methods: log, gamma, he, ahe, clahe, conversion")

def main():
    """
    Preprocess a full directory with a chosen method and save in a new directory
    """
    # Check usage
    if len(sys.argv) != 3:
        print("Usage: python preprocess.py input_directory preprocessing_method.")
        sys.exit(1)

    # Assign variables
    input_dir = sys.argv[1]
    method = sys.argv[2]

    # Assign output directory
    output_dir = f"{input_dir}_{method}"

    # Create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process images
    for root, _, files in os.walk(input_dir):

        # For each file
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.ppm')):
                img_path = os.path.join(root, file)
                img = load_img(img_path)
                img_array = img_to_array(img).astype('uint8')  # Convert to uint8 for OpenCV functions

                # Preprocess image
                preprocessed_img = preprocess_image(img_array, method)
                
                # Construct output path
                relative_path = os.path.relpath(root, input_dir)
                output_category_dir = os.path.join(output_dir, relative_path)
                if not os.path.exists(output_category_dir):
                    os.makedirs(output_category_dir)

                ### modify here
                output_path = os.path.join(output_category_dir, os.path.splitext(file)[0] + ".ppm") # modify for desired format

                # Save image
                save_img(output_path, preprocessed_img)
                print(f"Saved preprocessed image to {output_path}")

if __name__ == "__main__":
    main()
