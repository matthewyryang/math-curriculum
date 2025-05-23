import datasets
import os
import tqdm
import pandas as pd
import requests
import json
from tqdm import tqdm
from collections import defaultdict
# Load the dataset
ds = datasets.load_dataset('Asap7772/insight_evalsft_vllm', split='train')

reward_list = []
df = ds.to_pandas()

for i, row in tqdm(df.iterrows(), total=len(df)):
    # Ensure all values are strings
    num_responses = len(row['response'])
    
    insight_used = [str(x) for x in row['response']]
    paper1_prompt = [str(row['paper1_prompt'])] * num_responses
    paper2_prompt = [str(row['paper2_prompt'])] * num_responses
    joint_prompt = [str(row['joint_prompt'])] * num_responses
    no_context_prompt = [str(row['no_context_prompt'])] * num_responses
    
    request = {
        "paper1_examples": paper1_prompt,
        "paper2_examples": paper2_prompt,
        "joint_examples": joint_prompt,
        "no_context_examples": no_context_prompt,
        "insight_used": insight_used
    }
    
    # Verify the request is JSON serializable
    response = requests.post('http://localhost:8000/compute_contrastive_loss', json=request)
    curr_reward_list = response.json()

    reward_list.append(curr_reward_list)

reward_df = pd.DataFrame(reward_list)

# combine reward_df and df
df = pd.concat([df, reward_df], axis=1)
out_ds = datasets.Dataset.from_pandas(df)
out_ds.push_to_hub('Asap7772/insight_evalsft_vllm_reward')

print(f"Processed {len(reward_list)} requests successfully")