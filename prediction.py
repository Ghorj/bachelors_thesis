import sys
import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
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
        sys.exit("Usage: python prediction.py model_name prediction_folder_path")

    # Assign variables
    model_name = sys.argv[1]
    if not os.path.exists(model_name):
        sys.exit(f"Model file {model_name} does not exist.")
    prediction_folder_path = sys.argv[2]
    output_filename = model_name.split(".")[0] + "_prediction.png"
    preprocessing_method = "clahe"

    # Load the specified model
    model = tf.keras.models.load_model(model_name)
    print(f"{model_name} loaded.")

    # Predict images in the specified folder
    predict_images_in_folder(model, prediction_folder_path, output_filename, preprocessing_method)

def preprocess_image_wrapper(img_path, method=None):
    # Load the image
    img = load_img(img_path, target_size=(IMG_WIDTH, IMG_HEIGHT))

    # Apply specified preprocessing method
    if method:
        img = preprocess_image(img, method)

    # Normalize image to [0, 1] for model input
    img_array = img_to_array(img)
    img_array = img_array / 255.0

    img_array = np.expand_dims(img_array, axis=0)  # Create batch dimension
    return img_array

def predict_images_in_folder(model, folder_path, output_filename="predictions.png", method=None):
    class_labels = [
        "limite de velocidad 20 km/h", "limite de velocidad 30 km/h", "limite de velocidad 50 km/h", "limite de velocidad 60 km/h",
        "limite de velocidad 70 km/h", "limite de velocidad 80 km/h", "fin del limite de velocidad a 80", "limite de velocidad 100 km/h",
        "limite de velocidad 120 km/h", "no adelantar", "camiones no adelantar", "cruce", "paso con prioridad",
        "ceda el paso", "stop", "prohibido el paso", "prohibido el paso a camiones", "direccion prohibida", "precaucion", "curva a la izquierda",
        "curva a la derecha", "doble curva", "carretera con baches", "carretera resbaladiza", "carretera estrecha", "obras en carretera", "semaforo",
        "peatones", "ninos", "ciclistas", "nieve", "animales salvajes", "sin restriccion de velocidad",
        "giro obligatorio a la derecha", "giro obligatorio a la izquierda", "debe seguir de frente", "debe girar a la derecha o seguir de frente",
        "debe girar a la izquierda o seguir de frente",
        "conducir a la derecha del obstaculo", "conducir a la izquierda del obstaculo", "rotonda", "fin de las restricciones de adelantamiento",
        "camiones pueden adelantar"
    ]

    images = []
    labels = []
    img_files = [f for f in os.listdir(folder_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))][:25]

    # Iterate over the first 10 images in the folder
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