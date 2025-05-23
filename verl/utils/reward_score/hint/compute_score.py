"""
This module contains the RewardMathFn class, which evaluates mathematical answers
and assigns rewards based on their correctness. It utilizes a language model to 
validate answers when necessary.
"""
import datasets
from verl.utils.reward_score.curriculum_math.utils import extract_answer, grade_answer_sympy, grade_answer_mathd
from verl.utils.reward_score.hint.template import build_conv
from openai import OpenAI
import tenacity
import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

THOUGHT_DELIMITER_END = "</think>"
MODEL_NAME = "Qwen/Qwen3-4B"
SAMPLES_PER_HINT = 8
MAX_TOKENS = 8192
TEMPERATURE = 0.6

def create_client():
    return OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="token-abc123",
    )

@tenacity.retry(stop=tenacity.stop_after_attempt(3), wait=tenacity.wait_exponential(multiplier=1, min=4, max=15))
def get_single_completion(conv):
    client = create_client()
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=conv,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        n=1,
    )
    return completion.choices[0].message.content

def get_completions(conv):
    if SAMPLES_PER_HINT == 1:
        return get_single_completion(conv)
    
    with ProcessPoolExecutor(max_workers=SAMPLES_PER_HINT) as executor:
        futures = [executor.submit(get_single_completion, conv) for _ in range(SAMPLES_PER_HINT)]
        return [future.result() for future in as_completed(futures)]

def get_score(model_response, ground_truth):
    # # Extract solution.
    if THOUGHT_DELIMITER_END in model_response:
        model_solution = model_response.split(THOUGHT_DELIMITER_END)[1]
    else:
        return 0.
    model_solution = model_response
    
    model_answer = extract_answer(model_solution)
    if model_answer is None:
        return 0.

    # Process the ground truth(s)
    ground_truths = ground_truth

    # Convert single answer to list for uniform processing
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

    # Check against all possible correct answers
    for ground_truth in processed_ground_truths:
        is_correct = grade_answer_mathd(model_answer, ground_truth) or grade_answer_sympy(model_answer, ground_truth)
        if is_correct:
            return 1.

    return 0.

def compute_score(data_source, solution_str, ground_truth, extra_info):
    hint = solution_str # hint from the hint generator
    problem = extra_info['problem']
    conv = build_conv('cheatsheet', problem, hint)
    model_responses = get_completions(conv)
    scores = [get_score(model_response, ground_truth) for model_response in model_responses]
    avg_score = sum(scores) / (len(scores) or 1)
    return avg_score

if __name__ == '__main__':
    ds = datasets.load_dataset('Asap7772/dapo-hint-generator-qwen3-14b-filtered-lr1e6-0-5000', split='train')
    num_examples = 8
    
    def process_example(curr_row):
        hint = curr_row['all_hints']
        answer = curr_row['answer']
        
        print('--------------------------------')
        print(f'Problem: {curr_row["problem"]}')
        print(f'Hint: {hint}')
        print(f'Answer: {answer}')
        score = compute_score(
            data_source='math',
            solution_str=hint,
            ground_truth=answer,
            extra_info={
                'problem': curr_row['problem'],
            }
        )
        print(f'Score: {score}')
        print('--------------------------------')
        return score
    
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_example, ds[i]) for i in range(num_examples)]
        scores = []
        for future in tqdm.tqdm(as_completed(futures), total=8):
            scores.append(future.result())
    
    print(f'Average score across all examples: {sum(scores) / len(scores)}')