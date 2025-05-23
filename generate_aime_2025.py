import pandas as pd


from datasets import load_dataset
from datasets import concatenate_datasets


ds_1 = load_dataset("opencompass/AIME2025", "AIME2025-I")
ds_2 = load_dataset("opencompass/AIME2025", "AIME2025-II")


# Concatenate the datasets
ds = concatenate_datasets([ds_1['test'], ds_2['test']])


# Add the new column
def add_extra_info(example, idx):
    return {"extra_info": {"index": idx, "split": "aime2025"}}

def add_ability(example, idx):
    return {"ability": 'math'}

def add_reward_model(example, idx):
    return {"reward_model": {"ground_truth": example['answer'], "style": 'rule'}}

def add_datasource(example, idx):
    return {"data_source": 'aime2025'}

def add_prompt(example, idx):
    return {"prompt": [{"content": example['question'] + " Let's think step by step and output the final answer within \\boxed{}.", "role": "user"}]}

def add_level(example, idx):
    return {"level": "hard"}


ds = ds.map(add_extra_info, with_indices=True)
ds = ds.map(add_ability, with_indices=True)
ds = ds.map(add_reward_model, with_indices=True)
ds = ds.map(add_datasource, with_indices=True)
ds = ds.map(add_prompt, with_indices=True)
ds = ds.map(add_level, with_indices=True)


ds.to_parquet("/project/flame/asetlur/data/aime2025.parquet")