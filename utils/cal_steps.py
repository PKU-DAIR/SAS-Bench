# %%
import os
import json
import numpy as np

DIR = '/home/lpc/repos/SAS_Benchmark/datasets'
FILES = os.listdir(DIR)

for file_name in FILES:
    if len(file_name.split('_')) < 3:
        continue
    path = os.path.join(DIR, file_name)
    step_count = []
    length_count = []
    with open(path) as f:
        ori_data = f.readlines()
    ori_data = [json.loads(item) for item in ori_data]
    for item in ori_data:
        step_count.append(len(item['steps']))
        len_count = 0
        for step_item in item['steps']:
            len_count += len(list(step_item['response']))
        length_count.append(len_count)

    print(file_name, np.mean(step_count), np.mean(length_count))

# %%
