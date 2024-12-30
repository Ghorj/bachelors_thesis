import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from traffic import load_testing_data

def main():
    # Check usage
    if len(sys.argv) < 2:
        sys.exit("Usage: python evaluate.py model.keras")

    # load ANN model
    model_path = sys.argv[1]
    if not model_path.endswith(".keras"):
        sys.exit("Error: the file format must be .keras")
    
    
    # Data paths
    test_data_path = "GTSRB_new/Final_Test/Images_clahe"
    save_path = "Evaluation_folder"
    
    # if save folder doesnt exist, create it
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Evaluate model
    evaluate(model_path, test_data_path, save_path)
        
def evaluate(model_path, test_data_path, save_path):
    """
    Evaluate a trained model and save confusion matrix and F1-score
    """
    # Load data and labels
    model = load_model(model_path)
    test_data, y_true = load_testing_data(test_data_path)
    
    # Generate predictions
    y_pred_prob = model.predict(test_data)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    # F1-score
    f1_scores = f1_score(y_true, y_pred, average=None)
    
    # Plot and save confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predictions")
    plt.ylabel("True labels")
    plt.title("Confusion matrix")
    
    confusion_image_path = os.path.join(save_path, f"{os.path.splitext(model_path)[0]}_confusion_matrix.png")
    plt.savefig(confusion_image_path)
    plt.close()

    print(f"Confusion matrix saved in: {confusion_image_path}")
    
    # Plot and save F1-score
    plt.figure(figsize=(10, 7))
    plt.bar(range(len(f1_scores)), f1_scores, color="blue")
    plt.xlabel("Categories")
    plt.ylabel("F1-score")
    plt.title("F1-score per Category")
    plt.xticks(range(len(f1_scores)), [f"{i}" for i in range(len(f1_scores))], rotation=45)

    f1_score_image_path = os.path.join(save_path, f"{os.path.splitext(model_path)[0]}_f1_scores.png")
    plt.savefig(f1_score_image_path)
    plt.close() 

    print(f"F1-score saved in: {f1_score_image_path}")
    
    return conf_matrix, f1_scores

if __name__ == "__main__":
    main()
