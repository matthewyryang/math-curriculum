1. Follow instructions here: https://verl.readthedocs.io/en/latest/README_vllm0.8.html
2. `pip install seaborn`
3. `python examples/data_preprocess/math/generate_dataset.py --local_dir $local_dir --split $split`, where split is `easy`, `medium`, `hard`, `train`, or `test`. You should generate `test` and the split that you want to train on.
4. Adjust the variables accordingly and run `bash scripts/grpo/srun.sh`
