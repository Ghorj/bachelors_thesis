import sys
import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from preprocess import preprocess_image

# Image dimensions expected by the model
IMG_WIDTH, IMG_HEIGHT = 30, 30

# Preprocessing methods
METHODS = [
    "histogram_equalization",
    "clahe",
    "gamma",
    "log",
    "adaptive_histogram_equalization",
    ]

def main():
    """
    Predict photo categories and save in a file
    """
    # Check usage
    if len(sys.argv) < 3:
        sys.exit("Usage: python prediction.py model_name prediction_folder_path [output_file] [preprocessing_method]")

    # Assign variables
    model_name = sys.argv[1]
    if not os.path.exists(model_name):
        sys.exit(f"Model file {model_filename} does not exist.")
    prediction_folder_path = sys.argv[2]
    output_filename = sys.argv[3] if len(sys.argv) > 3 else "predictions.png"
    preprocessing_method = sys.argv[-1] if sys.argv[-1] in METHODS else None

    # Load the specified model
    model = tf.keras.models.load_model(model_name)
    print(f"{model_name} loaded.")

    # Predict images in the specified folder
    predict_images_in_folder(model, prediction_folder_path, output_filename, preprocessing_method)

def preprocess_image_wrapper(img_path, method=None):
    # Load the image
    img = image.load_img(img_path, target_size=(IMG_WIDTH, IMG_HEIGHT))
    img_array = image.img_to_array(img).astype("uint8")

    # Apply specified preprocessing method
    if method:
        img_array = preprocess_image(img_array, method)

    # Normalize image to [0, 1] for model input
    img_array = img_array / 255.0

    img_array = np.expand_dims(img_array, axis=0)  # Create batch dimension
    return img_array

def predict_images_in_folder(model, folder_path, output_filename="predictions.png", method=None):
    class_labels = [
        "maximum speed 20 km/h", "maximum speed 30 km/h", "maximum speed 50 km/h", "maximum speed 60 km/h",
        "maximum speed 70 km/h", "maximum speed 80 km/h", "no speed limitation", "maximum speed 100 km/h",
        "maximum speed 120 km/h", "do not overtake", "truck do not overtake", "crossroads ahead", "priority road",
        "give way", "stop", "no vehicles", "forbidden entrance for trucks", "no entry", "warning", "left turn",
        "right turn", "double curve", "rough road", "slippery road", "narrow road", "road works", "traffic lights ahead",
        "pedestrians", "children", "cyclists", "snow", "wild animals", "end of all overtaking and speed restrictions",
        "must turn right", "must turn left", "must go straight", "must go straight or turn right", "must go straight or turn left",
        "drive right side of the obstacle", "drive left side of the obstacle", "roundabout", "end of overtaking restrictions",
        "end of truck overtaking restrictions"
    ]

    images = []
    labels = []
    img_files = [f for f in os.listdir(folder_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))][:25]

    # Iterate over the first 25 images in the folder
    for img_file in img_files:
        img_path = os.path.join(folder_path, img_file)
        try:
            # Preprocess the image
            img_array = preprocess_image_wrapper(img_path, method)

            # Make prediction
            predictions = model.predict(img_array)

            # For classification models, get the index with the highest probability
            predicted_class = np.argmax(predictions, axis=1)[0]
            predicted_label = class_labels[predicted_class]

            # Load the original image
            original_img = cv2.imread(img_path)
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

            # Add image and label to lists
            images.append(original_img)
            labels.append(predicted_label)
            
            # Print the result
            print(f"Image: {img_file} | Predicted class index: {predicted_class} | Predicted label: {predicted_label}")

        except Exception as e:
            print(f"Error processing image {img_file}: {e}")

    # Create a plot with the images and predicted labels
    fig, axes = plt.subplots(5, 5, figsize=(15, 15))
    axes = axes.flatten()
    for img, label, ax in zip(images, labels, axes):
        ax.imshow(img)
        ax.set_title(label)
        ax.axis("off")

    plt.tight_layout()

    # Save plot in output file
    plt.savefig(output_filename)
    print(f"Predictions saved to {output_filename}")

if __name__ == "__main__":
    main()