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
parser.add_argument('--file_name', default='all', help='file name of the dataset without extension (e.g. 0_Physics_ShortAns), you can define `all` for all dataset files.')
parser.add_argument('--save_type_name', default='mimo', help='the prefix name of save dir (usually is the LLM name)')
parser.add_argument('--save_fig', default=0, help='whether save the figure')

if not cmd_args:
    args = parser.parse_args([]) # You can directly set above parameters in the default.
else:
    args = parser.parse_args()

SOURCE_DIR = os.path.join(args.file_dir + '_' + args.save_type_name + '_Scored')

labels = []
preds = []
step_labels = []
step_preds = []
if args.file_name != 'all':
    files = [args.file_name + '_prediction.jsonl']
else:
    files = os.listdir(SOURCE_DIR)
for file_name in files:
    if file_name.find('prediction') < 0:
        continue
    SOURCE_FILE = os.path.join(SOURCE_DIR, file_name)
    with open(SOURCE_FILE, encoding='utf-8') as f:
        ori_data = f.readlines()
    ori_data = [json.loads(line) for line in ori_data]
    for item in ori_data:
        total = float(item['total'])
        label = float(item['manual_label']) / total
        pred = float(item['pred_label']) / total
        labels.append(label)
        preds.append(pred)
        for step in item['steps']:
            step_labels.append(float(step['label']) / total)
        for step in item['pred_steps']:
            step_preds.append(float(step['step_score']) / total)

# %%
plt.hist([item for item in labels], range=(0, 1), bins=20, alpha=0.6, color='#FFC000', label='Gold')
plt.hist([item for item in preds], range=(0, 1), bins=20, alpha=0.6, color='#d3d3f9', label='Pred')
# plt.hist([item for item in step_labels], range=(0, 1), bins=20, alpha=0.3, color='#A6C9E8', label='Pred Step-wise Score')
# plt.hist([item for item in step_preds], range=(0, 1), bins=20, alpha=0.3, color='#44B6A3', label='Gold Step-wise Score')
plt.legend()
if str(args.save_fig) == '1':
    plt.savefig(f'{args.save_type_name}_pred_gold.svg', format='svg', dpi=300)

# %%
