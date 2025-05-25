from datasets import load_dataset
from verl.trainer.ppo.ray_trainer import r1_prompt_template
import argparse
from datasets import Dataset
from tqdm import tqdm
import random
import numpy as np
# Set the random seed for reproducibility
random.seed(42)
np.random.seed(42)
from transformers import AutoTokenizer, AutoModelForCausalLM
# from joblib import Parallel, delayed
import multiprocessing



def check_number_of_verifcation(text):
        steps = text.split('\n')
        cnt = 0
        for i, step in enumerate(steps):
            if step.startswith("Wait") or step.startswith("Let me verify") or step.startswith("But wait") or step.startswith("Alternatively") or step.startswith("Is there another way") or step.startswith("But let me double") or step.startswith("But hold on"):
                cnt += 1 
        return cnt
    


# def transform_record(dataset_record):
#     problem = dataset_record["conversations"][0]["value"]
#     completion = dataset_record["conversations"][1]["value"]
#     prompt = r1_prompt_template(problem)
#     completion = completion.replace('<|end_of_solution|>', '</answer>')
#     completion = completion.replace('<|begin_of_solution|>', '<answer>')
#     completion = completion.replace('<|begin_of_thought|>', '<think>')
#     completion = completion.replace('<|end_of_thought|>', '</think>')
#     return {
#         'prompt': prompt,
#         'completion': completion
#     }
    

def qwen_prompt_template(problem):
    try:
        problem = problem.split('your final response within \\boxed{}. ')[1]
        problem = problem + " Let's think step by step and output the final answer within \\boxed{}."
        return problem
    except:
        return None
    
def transform_record(dataset_record, tokenizer):
    problem = dataset_record["conversations"][0]["value"]
    completion = dataset_record["conversations"][1]["value"]
    
    problem = qwen_prompt_template(problem)
    if problem is None:
        return None
    prompt = [{"role": "user", "content": problem}]
    prompt = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
    
    completion = completion.replace('<|end_of_solution|>', '')
    completion = completion.replace('<|begin_of_solution|>', '')
    completion = completion.replace('<|begin_of_thought|>', '')
    completion = completion.replace('<|end_of_thought|>', '</think>')
    
    return {
        'prompt': prompt,
        'completion': completion
    }
    



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=120000, help="No. of samples")
    parser.add_argument("--max_length", type=int, default=26000, help="Max length of the prompt + completion")
    parser.add_argument("--min_length", type=int, default=16384, help="Min length of the prompt + completion")
    parser.add_argument("--save_location", type=str, default="/project/flame/asetlur/OpenThoughts-114k-qwen-format-minlen16k-maxlen24k", help="Save location for the transformed dataset")
    args = parser.parse_args()

    # Load the dataset
    ds = load_dataset("open-thoughts/OpenThoughts-114k", "default")
    filtered_ds = [ds['train'][idx] for idx in range(len(ds['train'])) if "python function" not in ds['train'][idx]['conversations'][0]['value'].lower()]
    print(f"Dataset size: {len(filtered_ds)}")
    total = len(filtered_ds)
    
    indices = list(range(total))
    random.shuffle(indices)
    top_indices = indices[:args.num_samples]
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    tokenizer.model_max_length = 32768
    
    # Transform the dataset
    transformed_ds = [
        transform_record(
             dataset_record=filtered_ds[idx], tokenizer=tokenizer) 
             for idx in tqdm(top_indices, total=len(top_indices))]
    transformed_ds = [record for record in transformed_ds if record is not None]
    


    def get_encoded_lengths(record, tokenizer):
        prompt_length = len(tokenizer.encode(record['prompt']))
        completion_length = len(tokenizer.encode(record['completion']))
        return prompt_length, completion_length

    pool = multiprocessing.Pool(processes=10)
    encoded_lengths = pool.starmap(get_encoded_lengths, [(record, tokenizer) for record in tqdm(transformed_ds, total=len(transformed_ds))])
    pool.close()
    pool.join()

    encoded_prompt_lengths = np.array([lengths[0] for lengths in encoded_lengths])
    encoded_completion_lengths = np.array([lengths[1] for lengths in encoded_lengths])

    selected_indices = np.where((encoded_prompt_lengths + encoded_completion_lengths < args.max_length) & (encoded_prompt_lengths + encoded_completion_lengths > args.min_length))[0] 


    selected_transformed_ds = [transformed_ds[i] for i in selected_indices]

    # Split the dataset into train and test sets
    train_size = int(0.95 * len(selected_transformed_ds))
    train_set = selected_transformed_ds[:train_size]
    test_set = selected_transformed_ds[train_size:]

    # Convert to a Dataset object
    transformed_ds_train = Dataset.from_list(train_set)
    transformed_ds_test = Dataset.from_list(test_set)

    print(f"Train set size: {len(transformed_ds_train)}")
    print(f"Test set size: {len(transformed_ds_test)}")

    # Save the transformed dataset as parquet
    transformed_ds_train.to_parquet(f'{args.save_location}/train.parquet')
    transformed_ds_test.to_parquet(f'{args.save_location}/test.parquet')


    import matplotlib.pyplot as plt
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    # Check the number of verifications
    verifications_train = [check_number_of_verifcation(record['completion']) for record in transformed_ds_train]
    verifications_test = [check_number_of_verifcation(record['completion']) for record in transformed_ds_test]
    # Plot a histogram
    ax1.hist(verifications_train, bins=100, alpha=0.5, label='Train Set', color='blue')
    ax1.hist(verifications_test, bins=100, alpha=0.5, label='Test Set', color='orange')
    ax1.set_xlabel('Number of Verifications')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Histogram of Verifications')
    

    
    # Encoded prompts
    encoded_prompt_lengths_train = [encoded_prompt_lengths[i] for i in range(len(encoded_prompt_lengths)) if i in selected_indices[:train_size]]
    encoded_prompt_lengths_test = [encoded_prompt_lengths[i] for i in range(len(encoded_prompt_lengths)) if i in selected_indices[train_size:]]
    # Print min, mean, and max length of train prompts
    print(f"Train Prompts: Min Length: {min(encoded_prompt_lengths_train)}, Mean Length: {sum(encoded_prompt_lengths_train) / len(encoded_prompt_lengths_train)}, Max Length: {max(encoded_prompt_lengths_train)}")
    # Print min, mean, and max length of test prompts
    print(f"Test Prompts - Min Length: {min(encoded_prompt_lengths_test)}, Mean Length: {sum(encoded_prompt_lengths_test) / len(encoded_prompt_lengths_test)}, Max Length: {max(encoded_prompt_lengths_test)}")
    # Plot the distribution of prompt lengths
    ax2.hist(encoded_prompt_lengths_train, bins=100, alpha=0.5, label='Train Set', color='blue')
    ax2.hist(encoded_prompt_lengths_test, bins=100, alpha=0.5, label='Test Set', color='orange')
    ax2.set_xlabel('Prompt Length')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Histogram of Prompt Lengths')
    ax2.legend()
    

    # Encode completions
    encoded_completion_lengths_train = [encoded_completion_lengths[i] for i in range(len(encoded_completion_lengths)) if i in selected_indices[:train_size]]
    encoded_completion_lengths_test = [encoded_completion_lengths[i] for i in range(len(encoded_completion_lengths)) if i in selected_indices[train_size:]]
    # Print min, mean, and max length of train completions
    print(f"Train Completions: Min Length: {min(encoded_completion_lengths_train)}, Mean Length: {sum(encoded_completion_lengths_train) / len(encoded_completion_lengths_train)}, Max Length: {max(encoded_completion_lengths_train)}")
    # Print min, mean, and max length of test completions
    print(f"Test Completions - Min Length: {min(encoded_completion_lengths_test)}, Mean Length: {sum(encoded_completion_lengths_test) / len(encoded_completion_lengths_test)}, Max Length: {max(encoded_completion_lengths_test)}")
    # Plot the distribution of completion lengths
    ax3.hist(encoded_completion_lengths_train, bins=100, alpha=0.5, label='Train Set', color='blue')
    ax3.hist(encoded_completion_lengths_test, bins=100, alpha=0.5, label='Test Set', color='orange')
    ax3.set_xlabel('Completion Length')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Histogram of Completion Lengths')
    ax3.legend()
    
    plt.savefig(f'openthoughts_sft_dataset.png', bbox_inches='tight')