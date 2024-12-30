import sys
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import time
import csv

# Constants
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.3
EPOCHS = 25
l2_strength = 0.01  # L2 regularization strength

def main():
    """
    Generate and train a custom neural network
    """
    # Check usage
    if len(sys.argv) < 2:
        sys.exit("Usage: python traffic.py data_directory [model.keras]")

    # Assign filenames
    data_dir = sys.argv[1]
    model_filename = sys.argv[2] if len(sys.argv) == 3 else None
    test_dir = "GTSRB_new/Final_Test/Images_clahe"  # modify accordingly

    # Split data
    images, labels = load_data(data_dir)
    test_images, test_labels = load_testing_data(test_dir)
    labels = tf.keras.utils.to_categorical(labels)
    test_labels = tf.keras.utils.to_categorical(test_labels)
    x_train = np.array(images)
    y_train = np.array(labels)
    x_test = np.array(test_images)
    y_test = np.array(test_labels)

    # Create an ImageDataGenerator for data augmentation
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False,  # Avoid horizontal flip for traffic signs
        fill_mode="nearest"
    )

    # Get a compiled neural network
    model = get_model()

    # Start timing the training process
    start_time = time.time()

    # Create early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",  
        patience=5,          # Number of epochs to wait after the last improvement
        restore_best_weights=True  
    )

    # Fit model on training data and save the history
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=32),
        epochs=EPOCHS,
        validation_data=(x_test, y_test),
        callbacks=[early_stopping],
        verbose=1
    )

    # End timing the training process
    total_training_time = time.time() - start_time
    print(f"Total training time: {total_training_time} seconds")

    # Evaluate neural network performance
    model.evaluate(x_test, y_test, verbose=2)

    # Save model to file if filename is provided
    if model_filename:
        model.save(model_filename)
        print(f"Model saved to {model_filename}")

        # Plot and save training accuracy
        plot_training_accuracy(history, model_filename)

def load_data(data_dir):
    """
    Load and preprocess images and labels
    """
    # initialize images and labels lists
    images = []
    labels = []

    # assign category values
    categories = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    # for each category
    for category in categories:

        # assign category directory and label
        category_dir = os.path.join(data_dir, category)
        label = int(category)

        # for each image file in the category directory
        for image_file in os.listdir(category_dir):

            # assign image path
            image_path = os.path.join(category_dir, image_file)

            
            try:
                # load image with target size
                img = load_img(image_path, target_size=(IMG_WIDTH, IMG_HEIGHT))

                # convert image to array if applicable and with values between 0 and 1
                img_array = img_to_array(img)
                img_array /= 255.0

                # append image and label to the list
                images.append(img_array)
                labels.append(label)
            
            except (IOError, SyntaxError, ValueError) as e:
                print(f"Skipping file {image_path}: {e}")
    
    # once loaded all images and labels return images and labels lists
    return images, labels

def load_testing_data(folder_path):
    """
    Load and preprocess images and labels for the validation data
    """
    # Initialize lists for images and labels
    images = []
    labels = {}
    
    # Path to the CSV file
    csv_file = os.path.join(folder_path, "GT-final_test.csv")
    
    # Read the CSV file with ";" as the delimiter
    with open(csv_file, mode="r") as file:
        csv_reader = csv.reader(file, delimiter=";")  # Specify delimiter
        next(csv_reader)  # Skip the header if there is one
        for row in csv_reader:
            if row:
                file_name = row[0]  # First column is the image file name
                label = int(row[-1])     # Last column is the label
                labels[file_name] = label
    
    # Get all .ppm image files in the folder and load them
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".ppm"):  # Looking for .ppm images
            image_path = os.path.join(folder_path, file_name)

            # Load the image using tensorflow.keras.preprocessing.image
            image = load_img(image_path, target_size=(IMG_WIDTH, IMG_HEIGHT))  # Load and resize image
            image_array = img_to_array(image) / 255.0  # Convert to array and normalize pixel values

            images.append(image_array)
    
    # Convert to numpy array
    images = np.array(images, dtype=np.float32)

    # Map the image paths with their labels
    image_labels = [labels[os.path.basename(img)] for img in os.listdir(folder_path) if img.endswith(".ppm")]

    return images, np.array(image_labels)


def get_model():
    """
    Create and compile a neural network
    """
    # Modify to match the desired neural network model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        
        # First convolutional block
        tf.keras.layers.Conv2D(14, (3, 3), padding="same", activation="relu",
                               kernel_regularizer=tf.keras.regularizers.l2(l2_strength)),
        tf.keras.layers.Conv2D(14, (3, 3), activation="relu",
                               kernel_regularizer=tf.keras.regularizers.l2(l2_strength)),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Second convolutional block
        tf.keras.layers.Conv2D(30, (3, 3), activation="relu",
                               kernel_regularizer=tf.keras.regularizers.l2(l2_strength)),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Flatten and fully connected layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.BatchNormalization(),    
        tf.keras.layers.Dense(208, activation="relu",
                               kernel_regularizer=tf.keras.regularizers.l2(l2_strength)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax",
                               kernel_regularizer=tf.keras.regularizers.l2(l2_strength))
    ])

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    return model

def plot_training_accuracy(history, model_filename):
    """
    Generate a file with the accuracy and validation accuracy data from training the neural network
    """
    # Assign variables
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    epochs = range(1, len(acc) + 1)

    # initialise plot
    plt.figure()

    # Training accuracy in blue
    plt.plot(epochs, acc, "b", label="Training accuracy")

    # Validation accuracy in red
    plt.plot(epochs, val_acc, "r", label="Validation accuracy")

    # Plot labels
    plt.title("Training and validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    # Save plot
    plt.savefig(f"{os.path.splitext(model_filename)[0]}_accuracy.png")
    plt.close()

if __name__ == "__main__":
    main()
