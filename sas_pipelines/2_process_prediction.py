# %%
import os
import re
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
parser.add_argument('--save_type_name', default='Qwen3_32B', help='the prefix name of save dir (usually is the LLM name)')

if not cmd_args:
    args = parser.parse_args([]) # You can directly set above parameters in the default.
else:
    args = parser.parse_args()

def unwrap_score_item(item, len_steps):
    item = json.loads(item)
    item = str(item)
    if item.find("{'total'") >= 0 and item.find('{"total"') < 0:
        item = item.replace('\'', '"')
    match_index = item.find('{"total"')
    if match_index >= 0:
        item = item[match_index:]
        item = item.replace('}```', '}')
        # Remove // Comments
        item = re.sub(r'//.*$', '', item, flags=re.MULTILINE)
        # Remove /* */ Comments
        item = re.sub(r'/\*[\s\S]*?\*/', '', item)
        item = json_repair.loads(item)
        if type(item) == list:
            item = item[0]
    else:
        item = {
            'total': 0,
            'pred_score': 0,
            'steps': [{'step_score': 0, 'errors': []} for _ in range(len_steps)]
        }
    
    if 'steps' not in item or type(item['steps']) != list:
        item['steps'] = [{'step_score': 0, 'errors': []} for _ in range(len_steps)]
    
    if 'pred_score' not in item:
        item['pred_score'] = 0
    
    for step_idx, step in enumerate(item['steps']):
        if type(step) != dict:
            item['steps'][step_idx] = {}
            step = item['steps'][step_idx]
        if 'step_score' not in step:
            step['step_score'] = 0
        try:
            step['step_score'] = int(step['step_score'])
        except:
            step['step_score'] = 0
        if 'errors' not in step or type(step['errors']) != list:
            step['errors'] = []
    return item
        
SOURCE_DIR = os.path.join(args.file_dir)
if args.file_name != 'all':
    files = [args.file_name + '.jsonl']
else:
    files = os.listdir(SOURCE_DIR)
for file_name in files:
    if len(file_name.split('_')) < 3:
        continue
    SOURCE_FILE = os.path.join(SOURCE_DIR, file_name)
    with open(SOURCE_FILE, encoding='utf-8') as f:
        ori_data = f.readlines()
    ori_data = [json.loads(line) for line in ori_data]

    SCORED_FILE = os.path.join(args.file_dir + '_' + args.save_type_name + '_Scored', f'{file_name.split(".jsonl")[0]}_scored.jsonl')
    print(SCORED_FILE)
    if not os.path.exists(SCORED_FILE):
        continue
    with open(SCORED_FILE, encoding='utf-8') as f:
        scored_data = f.readlines()

    for item, score_item in tqdm(zip(ori_data, scored_data)):
        score_item = score_item.split('\t')[1]
        len_steps = len(item['steps'])
        score_item = unwrap_score_item(score_item, len_steps=len_steps)
        try:
            item['pred_label'] = int(score_item['pred_score'])
        except:
            item['pred_label'] = 0
        item['pred_steps'] = score_item['steps']
        item['pred_steps'] = item['pred_steps'][:len_steps]
        if len(item['pred_steps']) < len_steps:
            for _ in range(len_steps - len(item['pred_steps'])):
                item['pred_steps'].append({'step_score': 0, 'errors': []})

    SAVE_DIR = os.path.join(os.path.dirname(SCORED_FILE), f'{file_name.split(".jsonl")[0]}_prediction.jsonl')
    with open(SAVE_DIR, 'w', encoding='utf-8') as f:
        for item in ori_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

# %%
