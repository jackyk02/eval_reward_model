import pickle
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
import json
from action_processing import ActionTokenizer
import requests
import json_numpy as json
import pickle

url = "http://127.0.0.1:3100/process"

# Load the Excel data
file_path = 'vla_comparisons_check.xlsx'
df = pd.read_excel(file_path)
df = df.head(640)

json_data = []

def tokenize_from_str(action):
    action = action.replace('\n', '')
    # Split by whitespace and filter out empty strings
    action_values = [x for x in action.strip(
        '[]').replace(',', ' ').split() if x]
    action_arr = np.array([float(x) for x in action_values])
    return action_arr

def get_rewards(instruction, image_path, actions):
    payload = {
        "instruction": instruction,
        "image_path": image_path,
        "action": actions
    }

    response = requests.post(url, data=json.dumps(payload))
    response = json.loads(response.text)
    rewards = response["rewards"]
    return rewards

with open("instruction_dict.pkl", "rb") as f:
    instruction_dict = pickle.load(f)

rewards_list = []  # List to store rewards for each row

for _, row in df.iterrows():
    if row['action0'] == 0:
        rewards_list.append(None)  # Append None if action0 is 0
        continue

    instruction = instruction_dict[int(row['index'])]
    image_path = f"/root/eval/images/000000{int(row['index'])}.jpg"
    actions = [tokenize_from_str(str(row['action1']))]

    try:
        rewards = get_rewards(instruction, image_path, actions)
        rewards_list.append(rewards)  # Append the rewards to the list
    except Exception as e:
        print(f"Error processing row {row['index']}: {e}")
        rewards_list.append(None)  # Append None if there's an error

# Add the rewards as a new column to the DataFrame
df['rewards'] = rewards_list

# Save the updated DataFrame to a new Excel file
df.to_excel('vla_comparisons_with_rewards.xlsx', index=False)
