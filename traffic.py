import sys
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import time

# Constants
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4
EPOCHS = 10

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

    # Split data
    images, labels = load_data(data_dir)
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Create an ImageDataGenerator for data augmentation
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False,  # Avoid horizontal flip for traffic signs
        fill_mode='nearest'
    )

    # Get a compiled neural network
    model = get_model()

    # Start timing the training process
    start_time = time.time()

    # Fit model on training data and save the history
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=32),
        epochs=EPOCHS,
        validation_data=(x_test, y_test),
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

            # load image with target size
            img = load_img(image_path, target_size=(IMG_WIDTH, IMG_HEIGHT))
            
            # convert image to array if applicable and with values between 0 and 1
            img_array = img_to_array(img)
            img_array /= 255.0

            # append image and label to the list
            images.append(img_array)
            labels.append(label)
    
    # once loaded all images and labels return images and labels lists
    return images, labels

def get_model():
    """
    Create and compile a neural network
    """
    # Modify to match the desired neural network model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        tf.keras.layers.Conv2D(256, (4, 4), activation="relu"),
        tf.keras.layers.MaxPooling2D((3, 3)),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Conv2D(256, (4, 4), activation="relu"),
        tf.keras.layers.MaxPooling2D((3, 3)),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Flatten(),       
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def plot_training_accuracy(history, model_filename):
    """
    Generate a file with the accuracy and validation accuracy data from training the neural network
    """
    # Assign variables
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(1, len(acc) + 1)

    # initialise plot
    plt.figure()

    # Training accuracy in blue
    plt.plot(epochs, acc, 'b', label='Training accuracy')

    # Validation accuracy in red
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')

    # Plot labels
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Save plot
    plt.savefig(f"{os.path.splitext(model_filename)[0]}_accuracy.png")
    plt.close()

if __name__ == "__main__":
    main()
