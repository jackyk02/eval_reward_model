import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define min_values and max_values
min_values = np.array([-0.07631552, -0.15209596, -0.15171302,
                       -0.22191394, -0.34210532, -0.73485305, 0.00000000])
max_values = np.array([0.08137930, 0.14595977, 0.14885315,
                       0.22793450, 0.20718527, 0.78006949, 1.00000000])
ranges = max_values - min_values

# Function to calculate NRMSE from NumPy arrays


def calculate_nrmse_array(groundtruth, action):
    normalized_diff = (groundtruth - action) / ranges
    nrmse = np.sqrt(np.mean(normalized_diff ** 2))
    return nrmse

# Function to calculate L1 Loss from NumPy arrays


def calculate_l1_loss_array(groundtruth, action):
    print(groundtruth)
    print("action: ", action)
    l1_loss = np.mean(np.abs(groundtruth - action))
    return l1_loss

# Function to clean and parse array-like strings


def clean_and_parse_array(array_str):
    # Remove extra spaces, brackets, and newlines, then split by comma or space
    cleaned_str = array_str.strip('[]').replace('\n', '').replace('  ', ' ')
    try:
        # Try parsing assuming comma-separated values
        if ',' in cleaned_str:
            parsed_array = np.array([float(x) for x in cleaned_str.split(',')])
        else:
            # Fallback for space-separated values
            parsed_array = np.array([float(x) for x in cleaned_str.split()])
    except ValueError:
        # Return an empty array if parsing fails
        parsed_array = np.array([])
    return parsed_array


# Load the Excel file
file_path = 'nrmse.xlsx'
data = pd.ExcelFile(file_path)

# Parse the relevant sheet
df = data.parse('Sheet1')

# Clean and parse array-like strings in the relevant columns
df['groundtruth_action'] = df['groundtruth_action'].apply(
    clean_and_parse_array)
df['random_action'] = df['random_action'].apply(clean_and_parse_array)
df['highest_score_action'] = df['highest_score_action'].apply(
    clean_and_parse_array)

# Filter rows with valid data (matching shapes)
df = df[df.apply(lambda row: row['groundtruth_action'].shape ==
                 row['random_action'].shape, axis=1)]

# Calculate NRMSE for random_action and highest_score_action
df['nrmse_random'] = df.apply(lambda row: calculate_nrmse_array(
    row['groundtruth_action'], row['random_action']), axis=1)
df['nrmse_highest'] = df.apply(lambda row: calculate_nrmse_array(
    row['groundtruth_action'], row['highest_score_action']), axis=1)

# Calculate L1 loss for random_action and highest_score_action
df['l1_loss_random'] = df.apply(lambda row: calculate_l1_loss_array(
    row['groundtruth_action'], row['random_action']), axis=1)
df['l1_loss_highest'] = df.apply(lambda row: calculate_l1_loss_array(
    row['groundtruth_action'], row['highest_score_action']), axis=1)

# Calculate averages
avg_nrmse_random = df['nrmse_random'].mean()
avg_nrmse_highest = df['nrmse_highest'].mean()
avg_l1_random = df['l1_loss_random'].mean()
avg_l1_highest = df['l1_loss_highest'].mean()

# Print results
print("Average NRMSE (Random Action):", avg_nrmse_random)
print("Average NRMSE (Highest Score Action):", avg_nrmse_highest)
print("Average L1 Loss (Random Action):", avg_l1_random)
print("Average L1 Loss (Highest Score Action):", avg_l1_highest)

# Visualization
labels = ['Random Action', 'Highest Score Action']
averages_nrmse = [avg_nrmse_random, avg_nrmse_highest]
averages_l1 = [avg_l1_random, avg_l1_highest]

# Plot Average NRMSE
plt.bar(labels, averages_nrmse, color='blue')
plt.title('Average NRMSE Comparison')
plt.ylabel('NRMSE')
plt.show()

# Plot Average L1 Loss
plt.bar(labels, averages_l1, color='green')
plt.title('Average L1 Loss Comparison')
plt.ylabel('L1 Loss')
plt.show()
