

import pandas as pd
import numpy as np
import json
import argparse
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from verl.utils.reward_score.curriculum_math.utils import extract_answer, grade_answer_sympy, grade_answer_mathd
from verl.utils.reward_score.curriculum_math.compute_score import compute_score
from collections import Counter
from itertools import combinations
    

THOUGHT_DELIMITER_END = "</think>"

def compute_pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def compute_maj_score(solutions, ground_truths):

    if isinstance(ground_truths, (str, float, int)):
            ground_truths = [ground_truths]        
    # Process each ground truth
    processed_ground_truths = []
    for truth in ground_truths:
        truth = str(truth)
        if "\\boxed" in truth:
            processed_truth = extract_answer(truth)
            if processed_truth is not None:
                processed_ground_truths.append(processed_truth)
        else:
            processed_ground_truths.append(truth)
    if not processed_ground_truths:
        return 0.
    extracted_solutions = []
    for solution_str in solutions:
        if THOUGHT_DELIMITER_END in solution_str:
            model_solution = solution_str.split(THOUGHT_DELIMITER_END)[1]
        else:
            model_solution = solution_str
        model_answer = extract_answer(model_solution)
        extracted_solutions.append(model_answer)
    counter = Counter(extracted_solutions)
    most_common_answer, most_common_count = counter.most_common(1)[0]
    # print(f"Most common answer: {most_common_answer}, count: {most_common_count}, ground_truth: {ground_truth}")
    if most_common_answer is not None:
        # Check against all possible correct answers
        for ground_truth in processed_ground_truths:
            is_correct = grade_answer_mathd(most_common_answer, ground_truth) or grade_answer_sympy(most_common_answer, ground_truth)
            if is_correct:
                return 1.0
    return 0.

if __name__ == "__main__":
    
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", help="Path to the JSON file")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Load the JSON file as a pandas DataFrame
    df = pd.read_json(args.filename)

    # tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    # tokenizer.model_max_length = 32768
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

    print(f"Dataset size: {len(df)}")
    print("Applying tokenization...")
    # Define a function to calculate the length of the prompt and completion
    def get_encoded_lengths(record):
        return len(tokenizer.encode(record['output']))
    
    # Apply the function to the DataFrame
    # df['len'] = df.apply(get_encoded_lengths, axis=1)

    q_df = pd.read_parquet('/project/flame/asetlur/data/hmmt_and_aime2025.parquet')
    q_df['index'] = q_df['extra_info'].transform(lambda x: x['index'])
    

    for source in set(df['source'].tolist()):
        print(f"Data source: {source}")
        source_df = df[df['source'] == source]
        print(f"Number of records: {len(source_df)}")
        print("Calculating scores...")
        for k in [1, 2, 4, 8, 16, 32]:
            pass_at_k_mean_score = 0.
            pass_at_k_mean_score_est2 = 0.
            maj_at_k_mean_score = 0.
            count = 0
            for idx, group in source_df.groupby("index"):
                # print(f"Scores for index {idx}: {len(scores)}, {sum(scores)}")
                scores = group['score'].tolist()
                count += 1
                pass_at_k_mean_score += compute_pass_at_k(len(scores), sum(scores), k)
                pass_at_k_mean_score_est2 += (1-np.power(1.0 - (sum(scores)/len(scores)),k))
                solutions = group['output'].tolist()
                # ground_truth = q_df[q_df['index']==idx].iloc[0]['reward_model']['ground_truth']
                # assert len(solutions) >= k
                # maj_correct = 0
                # total = 0
                # for i, subset in enumerate(combinations(solutions, k)):
                #     total += 1
                #     if total > 100:
                #         break
                #     maj_correct += compute_maj_score(subset, ground_truth)
                    # if scores[i] != compute_maj_score(subset, ground_truth):
                    #     print(subset, ground_truth, scores[i], compute_maj_score(subset, ground_truth))
                # maj_at_k_mean_score += (maj_correct / total)
            print(f"pass@{k}: {pass_at_k_mean_score / count:.4f}")
            # print(f"pass_est2@{k}: {pass_at_k_mean_score_est2 / count:.4f}")
            # print(f"maj@{k}: {maj_at_k_mean_score / count:.4f}")
        # source_df['under_2k'] = ((source_df['len'] < 2048) * (source_df['score'])).to_numpy(dtype=float)
        # source_df['under_4k'] = ((source_df['len'] < 4096) * (source_df['score'])).to_numpy(dtype=float)
        # source_df['under_8k'] = ((source_df['len'] < 8192) * (source_df['score'])).to_numpy(dtype=float)
        # source_df['under_16k'] = ((source_df['len'] < 16384) * (source_df['score'])).to_numpy(dtype=float)
        # source_df['under_24k'] = ((source_df['len'] < 24576) * (source_df['score'])).to_numpy(dtype=float)
        # source_df['under_32k'] = ((source_df['len'] < 32768) * (source_df['score'])).to_numpy(dtype=float)
        # Print the DataFrame
        # print(source_df.head())
        # print("2k:", source_df['under_2k'].mean())
        # print("4k:", source_df['under_4k'].mean())
        # print("8k:", source_df['under_8k'].mean())
        # print("16k:", source_df['under_16k'].mean())
        # print("24k:", source_df['under_24k'].mean())
        # print("32k:", source_df['under_32k'].mean())

    