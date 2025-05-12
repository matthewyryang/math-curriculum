from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--local_dir", type=str, required=True)
parser.add_argument("--hf_upload_path", type=str, required=True)
args = parser.parse_args()


model = AutoModelForCausalLM.from_pretrained(args.local_dir)
tokenizer = AutoTokenizer.from_pretrained(args.local_dir)

model.push_to_hub(args.hf_upload_path)
tokenizer.push_to_hub(args.hf_upload_path)