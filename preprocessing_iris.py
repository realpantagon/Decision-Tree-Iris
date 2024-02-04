import numpy as np
import csv

def calculate_mean_sd(data):
    iris_classes = np.unique(data[:, -1])  # Assuming the last column contains the class labels
    result = {}

    for iris_class in iris_classes:
        class_data = data[data[:, -1] == iris_class, :-1].astype(float)
        class_mean = np.mean(class_data, axis=0)
        class_sd = np.std(class_data, axis=0)

        result[iris_class] = {'mean': class_mean, 'sd': class_sd}

        # Print mean and standard deviation for each class
        print(f"Iris Class: {iris_class}")
        print(f"Mean: {class_mean}")
        print(f"Standard Deviation: {class_sd}")
        print()

    return result

def preprocess_data(data, mean_sd_dict, num_sd=2):
    filtered_data = []

    for row in data:
        iris_class = row[-1]
        features = row[:-1].astype(float)

        class_mean = mean_sd_dict[iris_class]['mean']
        class_sd = mean_sd_dict[iris_class]['sd']

        # Check if the features fall within mean ± num_sd * sd range
        if np.all(np.abs(features - class_mean) <= num_sd * class_sd):
            filtered_data.append(row)

    return np.array(filtered_data)

# Read data from iris.data file
with open('iris.data', 'r') as file:
    data_str = file.read()

# Parse the data into a NumPy array
data = np.array([line.split(',') for line in data_str.strip().split('\n')])

# Add header to the data
header = "sepal_length sepal_width petal_length petal_width class"
data_with_header = np.vstack([header.split(' '), data])

# Calculate mean and standard deviation for each class
mean_sd_dict = calculate_mean_sd(data_with_header[1:])

# Preprocess data by filtering out points outside mean ± 2 * sd range for each class
preprocessed_data = preprocess_data(data_with_header[1:], mean_sd_dict, num_sd=2)

# Write preprocessed data to CSV file with fixed-width spacing and header
np.savetxt('preprocessed_iris.csv', preprocessed_data, fmt='%-15s %-15s %-15s %-15s %s', delimiter=' ', header=header)

print("Preprocessed data has been written to 'preprocessed_iris.csv'.")


