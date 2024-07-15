import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

def main():
    """
    Calculate statistical values for a given dataset
    """
    # check usage
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python script_name.py dir1 [dir2]")
        sys.exit(1)
    
    # assign variables
    dir1 = sys.argv[1]
    dir2 = sys.argv[2] if len(sys.argv) == 3 else None

    # calculate statistics
    calculate_statistics(dir1, dir2)

def count_categories(directory):
    return len([name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))])

def calculate_statistics(dir1, dir2=None):
    # Determine the number of categories by counting folders in the first directory
    num_categories = count_categories(dir1)

    # Function to count images in a directory
    def count_images_in_directory(directory):
        counts = []
        for category in range(num_categories):
            category_dir = os.path.join(directory, str(category))
            if os.path.exists(category_dir):
                counts.append(len(os.listdir(category_dir)))
            else:
                counts.append(0)
        return counts

    # Get image counts for the first directory
    counts_dir1 = count_images_in_directory(dir1)

    # If a second directory is provided, get image counts and combine them
    if dir2:
        counts_dir2 = count_images_in_directory(dir2)
        combined_counts = np.array(counts_dir1) + np.array(counts_dir2)
    else:
        combined_counts = np.array(counts_dir1)

    # Calculate statistics
    count = len(combined_counts)
    mean = np.mean(combined_counts)
    median = np.median(combined_counts)
    mode = pd.Series(combined_counts).mode().values
    std_dev = np.std(combined_counts)
    variance = np.var(combined_counts)
    min_count = np.min(combined_counts)
    max_count = np.max(combined_counts)
    range_count = max_count - min_count
    q1 = np.percentile(combined_counts, 25)
    q3 = np.percentile(combined_counts, 75)
    iqr = q3 - q1
    cv = (std_dev / mean) * 100 if mean != 0 else 0
    skewness = skew(combined_counts)
    kurt = kurtosis(combined_counts)

    # Print statistics
    print(f"Count: {count}")
    print(f"Mean: {mean}")
    print(f"Median: {median}")
    print(f"Mode: {mode}")
    print(f"Standard Deviation: {std_dev}")
    print(f"Variance: {variance}")
    print(f"Minimum: {min_count}")
    print(f"Maximum: {max_count}")
    print(f"Range: {range_count}")
    print(f"Q1: {q1}")
    print(f"Q3: {q3}")
    print(f"IQR: {iqr}")
    print(f"Coefficient of Variation (CV): {cv}%")
    print(f"Skewness: {skewness}")
    print(f"Kurtosis: {kurt}")

if __name__ == "__main__":
    main()