from transformers import AutoModelForCausalLM, AutoTokenizer
import os
os.environ['HF_TOKEN'] = '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'

path = '/home/anikait.singh/rl_behaviors_verl_stable/sft/insight-warmstart-sft-qwen25-3b-3epoch-0501/global_step_30'
output_name = 'insight-warmstart-sft-qwen25-3b-3epoch-0501'

model = AutoModelForCausalLM.from_pretrained(path)
tokenizer = AutoTokenizer.from_pretrained(path)

model.push_to_hub(f'Asap7772/{output_name}')
tokenizer.push_to_hub(f'Asap7772/{output_name}')