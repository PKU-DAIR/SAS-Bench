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

if not cmd_args:
    args = parser.parse_args([]) # You can directly set above parameters in the default.
else:
    args = parser.parse_args()

SOURCE_DIR = os.path.join(args.file_dir + '_' + args.save_type_name + '_Scored')

from utils.collaborative_consistency_score import adjusted_qwk

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

    pred_result = []
    ori_result = []
    weights_result = []
    total_scores = []
    max_score = 0
    max_length = 0
    for item in tqdm(ori_data):
        if max_score < item['total']:
            max_score = item['total']
        total_scores.append(item['total'])
        ori_steps = item['steps']
        if 'pred_steps' not in item:
            pred_steps = [{"step_score": 0, "errors": []} for _ in ori_steps]
        else:
            pred_steps = item['pred_steps']
        ori_score = item['manual_label']
        if 'pred_label' not in item or item['pred_label'] == '':
            pred_score = 0
        else:
            pred_score = item['pred_label']
        ori_labels = [int(ori_score)]
        pred_labels = [int(pred_score)]
        weights = [0.5] + [0.5 / len(ori_steps) for _ in range(len(ori_steps))]
        for i in range(len(ori_steps)):
            try:
                ori_labels.append(int(ori_steps[i]['label']))
            except:
                ori_labels.append(0)
            if len(pred_steps) > i and type(pred_steps[i]) == dict and 'step_score' in pred_steps[i]:
                pred_labels.append(int(pred_steps[i]['step_score']))
            else:
                pred_labels.append(0)
        if max_length < len(ori_labels):
            max_length = len(ori_labels)
        pred_result.append(pred_labels)
        ori_result.append(ori_labels)
        weights_result.append(weights)

    # Padding
    for i in range(len(pred_result)):
        pred_result[i] += [0] * (max_length - len(pred_result[i]))
        ori_result[i] += [0] * (max_length - len(ori_result[i]))
        weights_result[i] += [0] * (max_length - len(weights_result[i]))
    max_scores = []
    for i in range(len(ori_result[0])):
        val_i = []
        for item in ori_result:
            val_i.append(item[i])
        for item in pred_result:
            val_i.append(item[i])
        max_scores.append(max(val_i))
    max_scores[0] = max_score

    all_norm_scores = []
    Ln = 1e-10
    for scores, max_s in zip(ori_result, total_scores):
        norm_score = scores[0] / (max_s + Ln)
        all_norm_scores.append(norm_score)
    range1, range2 = 0, 0
    sorted_scores = sorted(all_norm_scores)
    range1 = sorted_scores[int(len(sorted_scores) * 0.33)]
    range2 = sorted_scores[int(len(sorted_scores) * 0.67)]
    low_list = {
        'ori_result': [],
        'pred_result': [],
        'weights_result': [],
        'max_scores': max_scores
    }
    mid_list = {
        'ori_result': [],
        'pred_result': [],
        'weights_result': [],
        'max_scores': max_scores
    }
    high_list = {
        'ori_result': [],
        'pred_result': [],
        'weights_result': [],
        'max_scores': max_scores
    }
    for i in range(len(total_scores)):
        total = total_scores[i]
        gold = ori_result[i][0]
        gold = gold / total
        if gold <= range1:
            low_list['ori_result'].append(ori_result[i])
            low_list['pred_result'].append(pred_result[i])
            low_list['weights_result'].append(weights_result[i])
        elif gold < range2:
            mid_list['ori_result'].append(ori_result[i])
            mid_list['pred_result'].append(pred_result[i])
            mid_list['weights_result'].append(weights_result[i])
        else:
            high_list['ori_result'].append(ori_result[i])
            high_list['pred_result'].append(pred_result[i])
            high_list['weights_result'].append(weights_result[i])
    results.append((file_name, adjusted_qwk(*low_list.values())[0], adjusted_qwk(*mid_list.values())[0], adjusted_qwk(*high_list.values())[0]))

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

with open(f'{args.save_type_name}_range_ccs.csv', 'w+') as f:
    for item in results:
        cols = [item[0], str(round(item[1] * 100, 2)), str(round(item[2] * 100 ,2)), str(round(item[3] * 100 ,2))]
        f.write(','.join(cols) + '\n')

# %%
