import pickle
import pandas as pd
import numpy as np
import json
import requests
import random
from tqdm import tqdm

url = "http://127.0.0.1:3100/process"

# Load the Excel data
file_path = 'octo_data.xlsx'
df = pd.read_excel(file_path)

def act_to_array(action):
    action = action.replace('\n', '')
    # Split by whitespace and filter out empty strings
    action_values = [x for x in action.strip('[]').replace(',', ' ').split() if x]
    action_arr = np.array([float(x) for x in action_values])
    return action_arr.tolist()

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

# Load the instruction dictionary
with open("instruction_dict.pkl", "rb") as f:
    instruction_dict = pickle.load(f)

# Initialize lists to store results
results = []

# Group by index
for index, group in tqdm(df.groupby('index'), desc="Processing groups"):
    # Skip if action0 is 0
    if group.iloc[0]['action0'] == 0:
        continue

    # Extract instruction and image path
    instruction = instruction_dict[int(index)]
    image_path = f"/root/eval/images/000000{int(index)}.jpg"

    # Process actions in the group
    actions = [
        act_to_array(str(row['action1'])) for _, row in group.iterrows()
    ]

    # Initialize rewards list for all actions
    all_rewards = []

    # Process 4 actions at a time
    for i in range(0, len(actions), 4):
        action_subset = actions[i:i+4]
        rewards = get_rewards(instruction, image_path, action_subset)
        all_rewards.extend(rewards)

    # Find the action with the highest score
    highest_action_idx = np.argmax(all_rewards)

    # Randomly sample an action
    random_action_idx = random.randint(0, len(actions) - 1)

    # Append results
    results.append({
        "index": index,
        "groundtruth_action": group.iloc[0]['action0'],
        "random_action": actions[random_action_idx],
        "highest_score_action": actions[highest_action_idx]
    })

# Create a DataFrame from results
results_df = pd.DataFrame(results)

# Save to a new Excel file
results_file_path = 'processed_results.xlsx'
results_df.to_excel(results_file_path, index=False)

print(f"Results saved to {results_file_path}")