import pickle
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
import requests
import json
import json_numpy as json

url = "http://127.0.0.1:3100/process"

# Load the Excel data
file_path = 'preprocessed_64.xlsx'
df = pd.read_excel(file_path)

def clean_and_parse_array(array_str):
    # Remove extra spaces, brackets, and newlines, then split by comma or space
    cleaned_str = array_str.strip('[]').replace('\n', '').replace('  ', ' ')
    try:
        # Try parsing assuming comma-separated values
        if ',' in cleaned_str:
            parsed_array = [int(float(x)) for x in cleaned_str.split(',')]
        else:
            # Fallback for space-separated values
            parsed_array = [int(float(x)) for x in cleaned_str.split()]
    except ValueError:
        # Return an empty array if parsing fails
        parsed_array = []
    return parsed_array

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

# Load instruction dictionary
with open("../../data/instruction_dict_80k.pkl", "rb") as f:
    instruction_dict = pickle.load(f)

# Initialize new column for rewards
df['reward_action1'] = None

cnt = 0
correct = 0

for idx, row in df.iterrows():
    if row['action0'] == 0:
        continue
        
    instruction = instruction_dict[int(row['index'])]
    image_path = f"/root/eval/images/{int(row['index'])}.jpg"
    actions = [clean_and_parse_array(row['action1']), clean_and_parse_array(row['action1'])]

    rewards = get_rewards(instruction, image_path, actions)
    print(f"Processing row {idx}: {rewards}")
    
    # Store the reward for action1
    df.at[idx, 'reward_action1'] = rewards[0]

# Save the updated DataFrame to a new Excel file
output_file = 'preprocessed_64_with_rewards.xlsx'
df.to_excel(output_file, index=False)
print(f"\nProcessing complete. Updated file saved as: {output_file}")