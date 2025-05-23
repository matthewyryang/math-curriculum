from vllm import LLM, SamplingParams
import datasets

ds = datasets.load_dataset('Asap7772/250428_abstract_pair_rl', split='test')

num_samples = 100
if 0 < num_samples < len(ds):
    ds = ds.select(range(num_samples))

df = ds.to_pandas()

sampling_params = SamplingParams(temperature=0.6, n=8, max_tokens=4096)
model = LLM(model='/home/anikait.singh/rl_behaviors_verl_stable/sft/insight-sft-lr1e5-bsz64-maxlen4k/global_step_50')

all_responses = model.generate(df['query'], sampling_params)
all_responses = [[response.outputs[i].text for i in range(len(response.outputs))] for response in all_responses]

df['response'] = all_responses
ds = datasets.Dataset.from_pandas(df)
ds.push_to_hub('Asap7772/insight_evalsft_vllm')