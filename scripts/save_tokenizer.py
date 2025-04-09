from transformers import AutoTokenizer
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf_model_path', type=str, required=True, help="The path for the huggingface model")
    parser.add_argument('--target_dir', type=str, required=True, help="The path to save the tokenizer")
    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_path)
    tokenizer.save_pretrained(args.target_dir)
