import numpy as np
import matplotlib.pyplot as plt
import json
import os
import json
import pandas as pd
from transformers import AutoTokenizer

data_dir = '/project/flame/asetlur/easy-med-rollouts/'
json_files = [file for file in os.listdir(data_dir) if file.endswith('rollouts.json')]
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
tokenizer.model_max_length = 32768

data = {}
for file in json_files:
    iter = file.split('_')[0]
    iter = int(iter)
    if os.path.exists(os.path.join(data_dir, f"{iter}_processed.json")):
        continue
    print(f"Processing {file}...")
    df = pd.read_json(os.path.join(data_dir, file))
    print(f"Dataset size: {len(df)}")
    print("Applying tokenization...")
    # Define a function to calculate the length of the prompt and completion
    def get_encoded_lengths(record):
        return len(tokenizer.encode(record['output']))
    # Apply the function to the DataFrame
    df['len'] = df.apply(get_encoded_lengths, axis=1)
    # Save the DataFrame as a JSON file
    output_file = os.path.join(data_dir, f"{iter}_processed.json")
    df.to_json(output_file, orient='records')
    print(f"Processed data saved to {output_file}")
