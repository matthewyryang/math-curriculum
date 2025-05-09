import pandas as pd

dataset = pd.read_parquet("/project/flame/asetlur/data/aime.parquet")  # or load_dataset("json", data_files={"test": "your_file.json"})

# Add the new column
def add_extra_info(example, idx):
    return {"extra_info": {"index": idx, "split": split_name}}

def add_ability(example, idx):
    return {"ability": 'math'}

def add_reward_model(example, idx):
    return {"reward_model": {"ground_truth": example['answer'], "style": 'rule'}}

def add_datasource(example, idx):
    return {"data_source": example['source']}

def add_prompt(example, idx):
    return {"prompt": [{"content": example['prompt'] + " Let's think step by step and output the final answer within \\boxed{}.", "role": "user"}]}

def add_level(example, idx):
    return {"level": "hard"}

# Update the dataset dictionary
dataset['level'] = "hard"

dataset.to_parquet("/project/flame/asetlur/data/aime.parquet")