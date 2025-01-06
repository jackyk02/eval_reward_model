import pickle
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
import json
import requests
import numpy as np
import json_numpy as json
import pickle
url = "http://127.0.0.1:3100/process"

# Load the Excel data
file_path = 'validation_set.csv'
df = pd.read_csv(file_path)

json_data = []

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


with open("../data/instruction_dict_80k.pkl", "rb") as f:
    instruction_dict = pickle.load(f)

cnt = 0
correct = 0
for _, row in df.iterrows():
    if row['action0'] == 0:
        continue
    instruction = instruction_dict[int(row['index'])]
    image_path = f"/root/eval/images/{int(row['index'])}.jpg"
    actions = [clean_and_parse_array(row['action0']), clean_and_parse_array(row['action1'])]

    rewards = get_rewards(instruction, image_path, actions)
    print(rewards)
    win_action = 1
    if rewards[1] > rewards[0]:
        win_action = 2

    if win_action == row['winner']:
        correct += 1
    cnt += 1
    print("sucess rate: ", correct/cnt)