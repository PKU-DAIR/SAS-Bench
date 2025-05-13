# %%
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

plt.style.use('seaborn-v0_8-bright')

import sys
sys.path.append("../")
cmd_args = True

parser = ArgumentParser()
parser.add_argument('--file_dir', default='../datasets', help='the directory of the datasets.')
parser.add_argument('--file_name', default='10_Math_gapfilling', help='file name of the dataset, you should make sure it contains `test.jsonl` file')
parser.add_argument('--save_fig', default=0, help='whether save the figure')

if not cmd_args:
    args = parser.parse_args([]) # You can directly set above parameters in the default.
else:
    args = parser.parse_args()

SAMPLE_PATH = os.path.join(args.file_dir, args.file_name + '.jsonl')

with open(SAMPLE_PATH) as f:
    ori_data = f.readlines()
ori_data = [json.loads(item) for item in ori_data]

labels = []
steps = []
lengths = []
for item in ori_data:
    label = float(item['manual_label']) / float(item['total'])
    step_count = len(item['steps'])
    length = 0
    for step in item['steps']:
        length += len(step['response'])
    labels.append(label)
    steps.append(step_count)
    lengths.append(length)

def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


# %%
plt.hist([item for item in labels], range=(0, 1), bins=20, alpha=0.8, color='#FFC000', label='Label')
plt.hist(normalize(steps), range=(0, 1), bins=20, alpha=0.6, color='#d3d3f9', label='Steps')
# plt.hist(normalize(lengths), bins=20, alpha=0.3, color='#A6C9E8', label='Length')
plt.legend()
if str(args.save_fig) == '1':
    plt.savefig(f'{args.file_name}_distribution.svg', format='svg', dpi=300)

# %%
