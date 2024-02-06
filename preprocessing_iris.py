import numpy as np
import csv

def calculate_overall_statistics(data):
    # Convert all feature data to float, excluding the class labels
    features = data[:, :-1].astype(float)
    
    # Initialize result dictionary
    result = {}

    # Calculate statistics for each feature
    overall_mean = np.mean(features, axis=0)
    overall_sd = np.std(features, axis=0, ddof=0)  # Using ddof=0 for population standard deviation
    overall_min = np.min(features, axis=0)
    overall_max = np.max(features, axis=0)
    
    # Assigning calculated statistics to the result dictionary
    feature_names = ['sepal length', 'sepal width', 'petal length', 'petal width']
    for i, feature_name in enumerate(feature_names):
        result[feature_name] = {
            'Min': overall_min[i],
            'Max': overall_max[i],
            'Mean': overall_mean[i],
            'SD': overall_sd[i]
        }
    
    # Optionally, print out the statistics for each feature
    for feature in feature_names:
        print(f"{feature}:")
        print(f"  Min: {result[feature]['Min']}")
        print(f"  Max: {result[feature]['Max']}")
        print(f"  Mean: {result[feature]['Mean']}")
        print(f"  SD: {result[feature]['SD']}\n")
    
    return result

def preprocessing_data_with_overall_stats(data, stats_dict):
    processed_data = []
    for row in data:
        features = row[:-1].astype(float)
        new_row = []
        for i, feature in enumerate(features):
            # Use overall statistics for preprocessing
            feature_name = ['sepal length', 'sepal width', 'petal length', 'petal width'][i]
            feature_stats = stats_dict[feature_name]
            mean = feature_stats['Mean']
            sd = feature_stats['SD']

            # Abbreviations for the feature categories
            prefix = ['sl_', 'sw_', 'pl_', 'pw_'][i]
            
            if feature <= mean - sd:
                new_row.append(prefix + 's')
            elif mean - sd < feature <= mean + sd:
                new_row.append(prefix + 'm')
            else:
                new_row.append(prefix + 'l')
        
        # Append class label at the end
        iris_class = row[-1].strip()  # Ensure to strip whitespace
        new_row.append(iris_class)
        processed_data.append(new_row)
    return processed_data

# Read data from iris.data file and parse it
with open('iris.data', 'r') as file:
    data_str = file.read()
data = np.array([line.split(',') for line in data_str.strip().split('\n') if line])

# Calculate overall statistics
overall_stats = calculate_overall_statistics(data)

# Preprocess the data using overall statistics
preprocessed_data = preprocessing_data_with_overall_stats(data, overall_stats)

# Write the preprocessed data to CSV
with open('preprocessed_iris.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    for row in preprocessed_data:
        csvwriter.writerow(row)

# Write the preprocessed data to TXT with fixed-width formatting
with open('preprocessed_iris.txt', 'w') as txtfile:
    for row in preprocessed_data:
        txtfile.write(' '.join(row) + '\n')

print("Preprocessed data with overall statistics abbreviations has been written to both CSV and TXT files.")