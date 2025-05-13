# %%
import os
import json
import random
import json_repair
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

import sys
sys.path.append("../")
cmd_args = True

parser = ArgumentParser()
parser.add_argument('--file_dir', default='../datasets', help='the directory of the datasets.')
parser.add_argument('--file_name', default='all', help='file name of the dataset without extension (e.g. 0_Physics_ShortAns), you can define `all` for all dataset files.')
parser.add_argument('--save_type_name', default='Deepseek', help='the prefix name of save dir (usually is the LLM name)')
parser.add_argument('--skip_correct', default=True, help='batch size')

if not cmd_args:
    args = parser.parse_args([]) # You can directly set above parameters in the default.
else:
    args = parser.parse_args()

SKIP_CORRECT = args.skip_correct
SOURCE_DIR = os.path.join(args.file_dir + '_' + args.save_type_name + '_Scored')

# if it is english dataset, you may replace it with {'name': 'correct', 'description': 'the step is correct.'}
# but it depends on how you predefined the `name` of the correct step.
CORRECT_NAME = '步骤正确'
CORRECT_DESCRIPTION = '该步骤正确'

from utils.errors_consistency_score import compute_ecs

results = []
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

    ERROR_FILE = os.path.join(args.file_dir, 'error_type.jsonl')
    ID = file_name.split('_')[0]
    with open(ERROR_FILE, encoding='utf-8') as f:
        error_type_list = f.readlines()
    error_type_list = [json.loads(item) for item in error_type_list]
    error_type_item = []
    score_guideline = ''
    for item in error_type_list:
        if str(item['q_id']) == str(ID):
            error_type_item = item['errors']
            error_type_item.append({'name': CORRECT_NAME, 'description': CORRECT_DESCRIPTION})
            break

    error_to_id_dict = {}
    for i, item in enumerate(error_type_item):
        error_to_id_dict[item['name']] = i
    def err2idx(name):
        if name in error_to_id_dict:
            return error_to_id_dict[name]
        return len(error_to_id_dict) - 1

    pred_scores = []
    ori_scores = []

    total_scores = []

    pred_errors = []
    gold_errors = []
    max_error_length = len(error_type_item)
    max_length = 0
    for item in tqdm(ori_data):
        total_scores.append(item['total'])
        ori_score = item['manual_label']
        if 'pred_label' not in item or item['pred_label'] == '':
            pred_score = 0
        else:
            pred_score = item['pred_label']
        pred_scores.append(int(pred_score))
        ori_scores.append(int(ori_score))

        ori_steps = item['steps']
        if 'pred_steps' not in item:
            pred_steps = [{"step_score": 0, "errors": []} for _ in ori_steps]
        else:
            pred_steps = item['pred_steps']
        p_errors = [0 for _ in range(max_error_length)]
        g_errors = [0 for _ in range(max_error_length)]
        for step in ori_steps:
            errors = step['errors']
            for error in errors:
                if error == CORRECT_NAME and SKIP_CORRECT:
                    continue
                g_errors[err2idx(error)] += 1
        for step in pred_steps:
            errors = step['errors']
            for error in errors:
                if error == CORRECT_NAME and SKIP_CORRECT:
                    continue
                p_errors[err2idx(str(error))] += 1
        pred_errors.append(p_errors)
        gold_errors.append(g_errors)

    results.append((file_name, compute_ecs(pred_scores=pred_scores, ori_scores=ori_scores, total_scores=total_scores, pred_errors=pred_errors, gold_errors=gold_errors, max_error=max_error_length)))

# %%
def sort_func(x):
    try:
        x = x[0]
        x = x.split('_')[0]
        x = int(x)
        return x
    except:
        return 0

results = sorted(results, key=sort_func)
for item in results:
    print(item)

with open(f'{args.save_type_name}_ecs.csv', 'w+') as f:
    for item in results:
        cols = [item[0], str(round(item[1][0] * 100, 2)), str(round(item[1][1][0] * 100 ,2)), str(round(item[1][1][1] * 100 ,2)), str(round(item[1][1][2] * 100 ,2))]
        f.write(','.join(cols) + '\n')

# %%
