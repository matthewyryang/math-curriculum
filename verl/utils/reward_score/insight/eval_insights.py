"""
Script for evaluating document insights using contrastive learning.
Computes log probabilities of insights given different context conditions.
"""

import argparse
import os
from datetime import datetime
from typing import List, Tuple
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.multiprocessing as mp
from functools import partial
from concurrent.futures import ThreadPoolExecutor
import numpy as np


def construct_contrastive_pairs(
    documents: List[List[str]],
    insights: List[List[str]],
    pair_ids: List[int],
    score_prompt: str
):
    """
    Constructs different prompt variations for evaluating document insights.
    
    Args:
        documents: List of document pairs (each containing two documents)
        insights: List of insights for each document pair
        pair_ids: Unique identifier for each document pair
        score_prompt: Type of prompt to use ('none', 'generation', 'tags')
        
    Returns:
        Tuple containing prompts for paper1, paper2, joint contexts, 
        no-context baseline, pair IDs used, and processed insights
    """
    paper1_examples = []
    paper2_examples = []
    joint_examples = []
    no_context_examples = []
    pair_id_used = []
    insight_used = []
    
    for (doc1, doc2), insight_list, pair_id in zip(documents, insights, pair_ids):
        for insight in insight_list:
            if '[[' in insight or '<insight>' in insight:
                if score_prompt == 'none':
                    joint_text = f"Paper 1:{doc1}\nPaper 2:{doc2}\n"                    
                    paper1_text = f"Paper:{doc1}\n"
                    paper2_text = f"Paper:{doc2}\n"
                elif score_prompt == 'generation':
                    joint_text = (
                        f"Paper 1:\n"
                        f"{doc1}"
                        f"Paper 2:\n"
                        f"{doc2}"
                        "Your task is to identify and elaborate on an insight that only becomes apparent by combining information from both documents together—i.e., an insight that has high relevance when treating the documents jointly but low relevance if you were to consider each document alone. The insight should go beyond facts or conclusions that can already be inferred from either document by itself. "
                        "The insight should be self-contained. Write it in a way that doesn't require referencing where it came from. Do not mention the word \"paper\" or \"document\". "
                        "Let's think step by step. "
                    )
                    paper1_text = (
                        f"Paper:\n"
                        f"{doc1}"
                        "Your task is to identify and elaborate on an insight from the paper. "
                        "The insight should be self-contained. Write it in a way that doesn't require referencing where it came from. Do not mention the word \"paper\" or \"document\". "
                        "Let's think step by step. "
                    )
                    paper2_text = (
                        f"Paper:\n"
                        f"{doc2}"
                        "Your task is to identify and elaborate on an insight from the paper. "
                        "The insight should be self-contained. Write it in a way that doesn't require referencing where it came from. Do not mention the word \"paper\" or \"document\". "
                        "Let's think step by step. "
                    )
                elif score_prompt == 'tags':
                    joint_text = (
                        f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                        f"<|im_start|>user\n"
                        f"Paper 1:\n"
                        f"{doc1}"
                        f"Paper 2:\n"
                        f"{doc2}"
                        "Your task is to identify and elaborate on an insight that only becomes apparent by combining information from both documents together—i.e., an insight that has high relevance when treating the documents jointly but low relevance if you were to consider each document alone. The insight should go beyond facts or conclusions that can already be inferred from either document by itself. "
                        "The insight should be self-contained. Write it in a way that doesn't require referencing where it came from. Do not mention the word \"paper\" or \"document\"."
                        "<|im_end|>\n"
                        "<|im_start|>assistant\nLet's think step by step. "
                    )
                    paper1_text = (
                        f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                        f"<|im_start|>user\n"
                        f"Paper:\n"
                        f"{doc1}"
                        "Your task is to identify and elaborate on an insight from the paper. "
                        "The insight should be self-contained. Write it in a way that doesn't require referencing where it came from. Do not mention the word \"paper\" or \"document\"."
                        "<|im_end|>\n"
                        "<|im_start|>assistant\nLet's think step by step. "
                    )
                    paper2_text = (
                        f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                        f"<|im_start|>user\n"
                        f"Paper:\n"
                        f"{doc2}"
                        "Your task is to identify and elaborate on an insight from the paper. "
                        "The insight should be self-contained. Write it in a way that doesn't require referencing where it came from. Do not mention the word \"paper\" or \"document\"."
                        "<|im_end|>\n"
                        "<|im_start|>assistant\nLet's think step by step. "
                    )

                no_context_examples.append("Give me an insight. ")
                joint_examples.append(joint_text)
                paper1_examples.append(paper1_text)
                paper2_examples.append(paper2_text)
                pair_id_used.append(pair_id)
                
                # Extract the insight text from markup
                cleaned_insight = insight.replace('[[', '').replace(']]', '').replace('<insight>', '').replace('</insight>', '').strip()
                insight_used.append(cleaned_insight)
                
    return paper1_examples, paper2_examples, joint_examples, no_context_examples, pair_id_used, insight_used


def setup_model(model_name_or_path: str):
    """
    Sets up the tokenizer and model for inference.
    
    Args:
        model_name_or_path: Path or identifier for the model
        
    Returns:
        Tuple of (tokenizer, model)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    return tokenizer, model


def tokenize_batch(tokenizer, texts, max_length=None):
    """Parallel tokenization of a batch of texts."""
    with ThreadPoolExecutor() as executor:
        tokenize_fn = partial(tokenizer.encode, add_special_tokens=False)
        if max_length:
            tokenize_fn = partial(tokenize_fn, max_length=max_length, truncation=True)
        tokens = list(executor.map(tokenize_fn, texts))
    return tokens


def get_insight_log_prob(
    input_list: List[str],
    output_list: List[str],
    model,
    tokenizer,
    batch_size: int = 32,
    max_length: int = 4096,
    num_workers: int = os.cpu_count()
):
    """
    Tokenize and score all examples in batches, calculating log probabilities.
    
    Args:
        input_list: List of all prompts (context)
        output_list: List of insights to calculate log probs for
        model: Language model for scoring
        tokenizer: Tokenizer for the model
        batch_size: Batch size for processing
        max_length: Maximum sequence length for tokenization
        num_workers: Number of worker threads for parallel tokenization
        
    Returns:
        Tuple of tensors containing raw and length-normalized log probabilities
    """
    model.eval()
    
    # Pre-tokenize all inputs and outputs in parallel
    input_tokens = tokenize_batch(tokenizer, input_list)
    output_tokens = tokenize_batch(tokenizer, output_list)
    
    # Calculate max lengths for pre-allocation
    max_input_len = max(len(tokens) for tokens in input_tokens)
    max_output_len = max(len(tokens) for tokens in output_tokens)
    max_len = min(max_input_len + max_output_len, max_length)
    
    # Pre-allocate tensors for the entire dataset
    total_examples = len(input_list)
    input_ids = torch.full((total_examples, max_len), tokenizer.pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((total_examples, max_len), dtype=torch.long)
    
    # Fill tensors with tokenized data
    input_lens = []
    output_lens = []
    for i, (in_tokens, out_tokens) in enumerate(zip(input_tokens, output_tokens)):
        input_len = len(in_tokens)
        output_len = len(out_tokens)
        
        # Truncate if needed
        if input_len + output_len > max_length:
            in_tokens = in_tokens[:max_length - output_len]
            input_len = len(in_tokens)
        
        input_lens.append(input_len)
        output_lens.append(output_len)
        
        # Fill tensors
        input_ids[i, :input_len] = torch.tensor(in_tokens)
        input_ids[i, input_len:input_len + output_len] = torch.tensor(out_tokens)
        attention_mask[i, :input_len + output_len] = 1
    
    # Move tensors to device
    input_ids = input_ids.to(model.device)
    attention_mask = attention_mask.to(model.device)
    
    # Process in batches
    log_sums = []
    log_sums_avg = []
    
    with torch.no_grad():
        for i in range(0, total_examples, batch_size):
            batch_end = min(i + batch_size, total_examples)
            batch_input_ids = input_ids[i:batch_end]
            batch_attention_mask = attention_mask[i:batch_end]
            
            # Get model outputs
            outputs = model(batch_input_ids, attention_mask=batch_attention_mask)
            logits = outputs.logits  # (batch_size, seq_len, vocab_size)
            
            # Process each example in the batch
            for j in range(batch_end - i):
                idx = i + j
                input_len = input_lens[idx]
                output_len = output_lens[idx]
                
                # Get relevant logits and tokens efficiently
                relevant_logits = logits[j, input_len-1:input_len+output_len-1]
                relevant_tokens = batch_input_ids[j, input_len:input_len+output_len]
                
                # Calculate log probabilities in one operation
                log_probs = F.log_softmax(relevant_logits, dim=-1)
                token_log_probs = log_probs.gather(1, relevant_tokens.unsqueeze(1)).squeeze(1)
                
                # Sum log probabilities
                log_sum = token_log_probs.sum().item()
                log_sums.append(log_sum)
                log_sums_avg.append(log_sum / float(max(output_len, 1)))
    
    return torch.tensor(log_sums), torch.tensor(log_sums_avg)


def compute_contrastive_loss(
    paper1_examples, 
    paper2_examples, 
    joint_examples, 
    no_context_examples, 
    insight_used, 
    model, 
    tokenizer,
    batch_size: int = 32,
    contrastive_loss_type: str = 'logsumexp'
):
    """
    Compute log probabilities and contrastive loss for insights under different contexts.
    
    Args:
        paper1_examples: Prompts with only paper 1 context
        paper2_examples: Prompts with only paper 2 context
        joint_examples: Prompts with both papers as context
        no_context_examples: Baseline prompts with no paper context
        insight_used: The insights to calculate probabilities for
        model: Language model
        tokenizer: Tokenizer for the model
        batch_size: Batch size for processing
        
    Returns:
        Tuple of various scores and the contrastive loss
    """
    # Combine all contexts and insights into single lists for batch processing
    all_contexts = paper1_examples + paper2_examples + joint_examples + no_context_examples
    all_insights = insight_used * 4  # Repeat insights for each context type
    
    # Get scores for all contexts in a single batch
    all_scores, all_scores_avg = get_insight_log_prob(
        all_contexts, all_insights, model, tokenizer, batch_size=batch_size
    )
    
    # Split scores back into their respective contexts using efficient tensor operations
    n = len(paper1_examples)
    scores = all_scores.view(4, n)  # Reshape into (4, n) tensor
    scores_avg = all_scores_avg.view(4, n)  # Reshape into (4, n) tensor
    
    # Extract scores for each context type
    paper1_scores, paper2_scores, joint_scores, no_context_scores = scores
    paper1_scores_avg, paper2_scores_avg, joint_scores_avg, no_context_scores_avg = scores_avg
    
    if contrastive_loss_type == 'max':
        max_score = torch.max(torch.max(paper1_scores, paper2_scores), no_context_scores)
        max_score_avg = torch.max(torch.max(paper1_scores_avg, paper2_scores_avg), no_context_scores_avg)
    elif contrastive_loss_type == 'logsumexp':
        max_score = torch.logsumexp(torch.stack([paper1_scores, paper2_scores, no_context_scores]), dim=0)
        max_score_avg = torch.logsumexp(torch.stack([paper1_scores_avg, paper2_scores_avg, no_context_scores_avg]), dim=0)
    elif contrastive_loss_type == 'sum':
        max_score = paper1_scores + paper2_scores + no_context_scores
        max_score_avg = paper1_scores_avg + paper2_scores_avg + no_context_scores_avg
    else:
        raise ValueError(f"Invalid contrastive loss type: {contrastive_loss_type}")
    
    # Calculate contrastive loss using vectorized operations
    contrastive_loss = joint_scores - max_score
    contrastive_loss_avg = joint_scores_avg - max_score_avg
    
    return (
        paper1_scores, 
        paper1_scores_avg, 
        paper2_scores, 
        paper2_scores_avg, 
        joint_scores, 
        joint_scores_avg, 
        no_context_scores, 
        no_context_scores_avg, 
        contrastive_loss,
        contrastive_loss_avg
    )


def main():
    """Main function to run the evaluation process."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluate insights using contrastive learning")
    parser.add_argument('--model_id', type=str, help="Model identifier or path")
    parser.add_argument('--score_prompt', type=str, help="Type of prompt to use")
    parser.add_argument('--insights_path', type=str, help="Path to insights data")
    parser.add_argument('--result_folder_prefix', type=str, help="Prefix for results folder")
    args = parser.parse_args()

    MODEL_ID = args.model_id
    print("\nMODEL_ID:", MODEL_ID)

    SCORE_PROMPT = args.score_prompt
    INSIGHTS_PATH = args.insights_path
    
    # Create output filename from inputs
    insights_suffix = INSIGHTS_PATH.split('/')[-2] + '_' + INSIGHTS_PATH.split('/')[-1].split('.json')[0]
    RESULT_FOLDER = os.path.join(
        args.result_folder_prefix,
        f"{MODEL_ID.split('/')[-1]}_eval_{insights_suffix}_scoreprompt_{SCORE_PROMPT}.jsonl"
    )
   
    # Create output directory if it doesn't exist
    os.makedirs(args.result_folder_prefix, exist_ok=True)

    # Load insights data
    print('Loading insights:', datetime.now())
    insights_df = pd.read_json(INSIGHTS_PATH, lines=True)
    abstracts = list(insights_df['abstract'])
    insights = list(insights_df['insight'])
    
    # Get pair IDs if available, otherwise create sequential IDs
    if 'pair_id' in insights_df.columns:
        pair_ids = list(insights_df['pair_id'])
    else:
        pair_ids = list(range(len(insights_df)))

    assert len(abstracts) == len(insights)
    
    # Prepare evaluation data
    paper1_examples, paper2_examples, joint_examples, no_context_examples, pair_id_used, insight_used = (
        construct_contrastive_pairs(abstracts, insights, pair_ids, SCORE_PROMPT)
    )
    
    # Set up model and tokenizer
    tokenizer, model = setup_model(MODEL_ID)
    print('Done constructing:', datetime.now())
    
    # Process in batches
    results_all = pd.DataFrame()
    BATCH_SIZE = 100
    
    for e in range(0, len(paper1_examples), BATCH_SIZE):
        batch_paper1_examples = paper1_examples[e:e+BATCH_SIZE]
        batch_paper2_examples = paper2_examples[e:e+BATCH_SIZE]
        batch_joint_examples = joint_examples[e:e+BATCH_SIZE]
        batch_no_context_examples = no_context_examples[e:e+BATCH_SIZE]  
        batch_insight_used = insight_used[e:e+BATCH_SIZE]
        
        # Compute scores
        (   paper1_scores, 
            paper1_scores_avg, 
            paper2_scores, 
            paper2_scores_avg, 
            joint_scores, 
            joint_scores_avg, 
            no_context_scores, 
            no_context_scores_avg, 
            contrastive_loss
        ) = compute_contrastive_loss(
            batch_paper1_examples, 
            batch_paper2_examples, 
            batch_joint_examples, 
            batch_no_context_examples, 
            batch_insight_used,
            model, 
            tokenizer
        )
    
        # Store results
        results = pd.DataFrame({
            'pair_id': pair_id_used[e:e+BATCH_SIZE],
            'insight': batch_insight_used,
            'paper1_scores': paper1_scores.cpu().numpy(),
            'paper2_scores': paper2_scores.cpu().numpy(),
            'joint_scores': joint_scores.cpu().numpy(),
            'no_context_scores': no_context_scores.cpu().numpy(),
            'contrastive_loss': contrastive_loss.cpu().numpy(),
            'paper1_scores_avg': paper1_scores_avg.cpu().numpy(),
            'paper2_scores_avg': paper2_scores_avg.cpu().numpy(),
            'joint_scores_avg': joint_scores_avg.cpu().numpy(),
            'no_context_scores_avg': no_context_scores_avg.cpu().numpy(),
        })
        
        # Append batch results to all results
        results_all = pd.concat([results_all, results])
        
        # Save intermediate results
        results_all.to_json(RESULT_FOLDER, orient="records", lines=True)
        
        print(f"Completed batch starting at index {e}")
        print(datetime.now())
        
    print('Done:', datetime.now())


if __name__ == "__main__":
    main()