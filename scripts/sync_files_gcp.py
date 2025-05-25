import os
from multiprocessing import Pool

def sync_directory(path):
    if os.path.isdir(path):
        if path.strip() in restricted_dirs:
            return
        print(f"Syncing {path}")
        command = f"gsutil -m cp -rn {path} gs://anikait-rlhf-central2/rl_behaviors_verl_stable/"
        os.system(command)

base_path = "/home/anikait.singh/rl_behaviors_verl_stable"
all_files = os.listdir(base_path)
restricted_dirs = ['outputs', 'ppo', 'sft', 'sft_hdfs', 'wandb', 'outputs']

# Create a pool of workers and map the sync function to all directories
with Pool(processes=os.cpu_count()) as pool:
    pool.map(sync_directory, all_files)

