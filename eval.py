import pickle
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
import json
from action_processing import ActionTokenizer
import requests
import numpy as np
import json_numpy as json
import pickle
url = "http://127.0.0.1:3100/process"

# Load the Excel data
file_path = 'validation_2.xlsx'
df = pd.read_excel(file_path)

json_data = []

def tokenize_from_str(action):
    action = action.replace('\n', '')
    # Split by whitespace and filter out empty strings
    action_values = [x for x in action.strip(
        '[]').replace(',', ' ').split() if x]
    action_arr = np.array([float(x) for x in action_values])
    # print(action_arr)
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

cnt = 0
correct = 0
for _, row in df.iterrows():
    if row['action0'] == 0:
        continue
    instruction = instruction_dict[int(row['index'])]
    image_path = f"/root/eval/images/000000{int(row['index'])}.jpg"
    actions = [tokenize_from_str(str(row['action0'])), tokenize_from_str(str(row['action1']))]

    rewards = get_rewards(instruction, image_path, actions)
    win_action = 1
    if rewards[1] > rewards[0]:
        win_action = 2

    if win_action == row['winner']:
        correct += 1
    cnt += 1
    print("sucess rate: ", correct/cnt)


