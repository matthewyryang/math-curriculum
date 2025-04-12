"""
This module contains the RewardMathFn class, which evaluates mathematical answers
and assigns rewards based on their correctness. It utilizes a language model to 
validate answers when necessary.
"""
from verl.utils.reward_score.curriculum_math.utils import extract_answer, grade_answer_sympy, grade_answer_mathd

THOUGHT_DELIMITER_END = "</think>"


def compute_score(data_source, solution_str, ground_truth, extra_info):


        model_response = solution_str
        
        # # Extract solution.
        # if THOUGHT_DELIMITER_END in model_response:
        #     model_solution = model_response.split(THOUGHT_DELIMITER_END)[1]
        # else:
        #     return 0.
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
