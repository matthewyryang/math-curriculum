import os
from datasets import load_dataset, concatenate_datasets
from typing import Dict, List, Optional, Any
import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='/home/asetlur/math-curriculum/data')
    parser.add_argument('--remote_dir', default='d1shs0ap/math')
    parser.add_argument('--split', default=['test'], nargs='+')
    

    args = parser.parse_args()

    datasets_to_load = args.split
    datasets = []

    for dataset_name in datasets_to_load:
        datasets.append(load_dataset(args.remote_dir, split=dataset_name))

    dataset = concatenate_datasets(datasets)
        
    print(f"Loaded dataset with {len(dataset)} examples.")

    def make_map_fn(split: str):
        """Create a mapping function to process dataset examples.

        Args:
            split: Dataset split name ('train' or 'test')

        Returns:
            Function that processes individual dataset examples
        """
        def process_fn(example: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
            question = example.pop('problem')
            instruction = "Let's think step by step and output the final answer within \\boxed{}."
            question = f"{question} {instruction}"
            answer = example.pop('answer')

            data = {
                "data_source": "",
                "prompt": [{
                    "role": "user",
                    "content": question
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer
                },
                "extra_info": {
                    'split': split,
                    'index': idx
                }
            }
            return data
        return process_fn
    
    dataset = dataset.map(function=make_map_fn(args.split), with_indices=True)

    dataset.to_parquet(os.path.join(args.local_dir, f'{"_".join(args.split)}.parquet'))