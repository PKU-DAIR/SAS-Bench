# %%
import os
import json
import random
import json_repair
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from argparse import ArgumentParser

import sys
sys.path.append("../")
cmd_args = True
# Add params n_gpu
parser = ArgumentParser()
parser.add_argument('--n_gpu', default=0, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--file_dir', default='../datasets', help='the directory of the datasets.')
parser.add_argument('--file_name', default='9_English_gapfilling', help='file name of the dataset without extension (e.g. 0_Physics_ShortAns)')
parser.add_argument('--llm_name', default='', help='the prefix name of save dir (usually is the LLM name)')
parser.add_argument('--save_type_name', default='GLM4', help='the prefix name of save dir (usually is the LLM name)')
parser.add_argument('--few_shot_num', default=0, help='decide the number of few-shot samples')
parser.add_argument('--use_guideline', default='1', help='whether use scoring guideline')
parser.add_argument('--model_from_pretrained', default='/home/lpc/models/glm-4-9b-chat/', help='model from pretrained')
parser.add_argument('--vllm', default='0', help='whether use vllm')
parser.add_argument('--tensor_parallel_size', default=1, help='tensor_parallel_size (TP) for vLLM')
parser.add_argument('--max_new_tokens', default=1024, help='max new tokens')
parser.add_argument('--do_sample', default='0', help='do_sample, useless for vLLM')
parser.add_argument('--temperature', default=0.6, help='temperature, if temperture > 0, it will work on vLLM.')
parser.add_argument('--top_p', default=0.95, help='top_p, if top_p < 1.0, it will work on vLLM')
parser.add_argument('--skip_thinking', default='0', help='skip deep thinking in RL model with <think>\n\n</think>')
parser.add_argument('--batch_size', default=5, help='batch size, suggest to set it larger when use vLLM')
parser.add_argument('--fix_reasoning', default=1, help='Re-generate with longer length for the result without finishing thinking')

if not cmd_args:
    args = parser.parse_args([]) # You can directly set above parameters in the default.
else:
    args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.n_gpu)

API_MODELS = ['gpt-4o-mini', 'deepseek-chat', 'deepseek-reasoner']
API_CONFIGS = [('OpenAI', None), ('Deepseek', 'https://api.deepseek.com'), ('Deepseek', 'https://api.deepseek.com')]

USE_VLLM = str(args.vllm) == '1'

llm_name = args.llm_name if args.llm_name != '' else args.save_type_name
if llm_name == 'GLM3':
    from main.predictor.chatglm import Predictor
elif llm_name in API_MODELS:
    from main.predictor.openai import Predictor
elif USE_VLLM:
    from main.predictor.vllm import Predictor
else:
    from main.predictor.llm import Predictor

if llm_name not in API_MODELS:
    pred = Predictor(model_from_pretrained=args.model_from_pretrained, tensor_parallel_size=int(args.tensor_parallel_size))
else:
    CONFIG_INDEX = API_MODELS.index(llm_name)
    with open('api_key.txt') as f:
        api_keys = f.readlines()
    for key_item in api_keys:
        key_item = key_item.strip().split(' ')
        if len(key_item) == 1:
            api_key = key_item
            break
        else:
            if key_item[0] == API_CONFIGS[CONFIG_INDEX][0]:
                api_key = key_item[1]
                break
    pred = Predictor(api_key=api_key, base_url=API_CONFIGS[CONFIG_INDEX][1])

# %%
SOURCE_FILE = os.path.join(args.file_dir, f'{args.file_name}.jsonl')
ERROR_TYPE_FILE = os.path.join(args.file_dir, 'error_type.jsonl')
SAVE_DIR = os.path.dirname(SOURCE_FILE) + f'_{args.save_type_name}_Scored'
basename = os.path.basename(SOURCE_FILE)
SAVE_FILE = os.path.join(SAVE_DIR,
                         basename.split('.')[0]+'_scored.jsonl')
FEW_SHOT_NUM = int(args.few_shot_num)
BATCH_SIZE = int(args.batch_size)
MAX_NEW_TOKENS = int(args.max_new_tokens)

# if it is english dataset, you may replace it with {'name': 'correct', 'description': 'the step is correct.'}
# but it depends on how you predefined the `name` of the correct step.
CORRECT_NAME = '步骤正确'
CORRECT_DESCRIPTION = '该步骤正确'
PREDICT_PROMPT = ''
FEW_SHOT_PROMPT = {'prefix': '', 'suffix': '', 'question': '', 'reference': '', 'total': '', 'analysis': '', 'student_answer': '', 'output': ''}

with open('../prompts/predict.txt') as f:
    PREDICT_PROMPT = f.read().strip()

with open('../prompts/few_shot_prompt.json') as f:
    FEW_SHOT_PROMPT = json.load(f)

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

ID = args.file_name.split('_')[0]
with open(ERROR_TYPE_FILE, encoding='utf-8') as f:
    error_type_list = f.readlines()
error_type_list = [json.loads(item) for item in error_type_list]
error_type_item = []
score_guideline = ''
for item in error_type_list:
    if str(item['q_id']) == str(ID):
        score_guideline = item['guideline']
        error_type_item = item['errors']
        error_type_item.append({'name': CORRECT_NAME, 'description': CORRECT_DESCRIPTION})
        break

if str(args.use_guideline) != '1':
    score_guideline = ''

# Read the JSON file
with open(SOURCE_FILE, encoding='utf-8') as f:
    ori_data = f.readlines()
ori_data = [json.loads(item) for item in ori_data]

ori_data_id_dict = {}
for idx, item in enumerate(ori_data):
    if item['id'] not in ori_data_id_dict:
        ori_data_id_dict[item['id']] = idx

if str(args.fix_reasoning) == '1' and os.path.exists(SAVE_FILE):
    with open(SAVE_FILE) as f:
        save_data = f.readlines()
    
    count = 0
    for item in save_data:
        item = item.split('\t')
        id, content = item[0], item[1]
        if id in ori_data_id_dict:
            idx = ori_data_id_dict[id]
            content = json.loads(content.strip())
            if type(content) != str:
                content = str(content)
            if content.find('{"total"') >= 0 or content.find('{\'total\'') >= 0:
                ori_data[idx]['cache'] = content.strip()
                count += 1
    print(f'Found correct generation: {count}')

few_shot_prompt = ''
if FEW_SHOT_NUM > 0:
    from utils.load_few_shot import get_few_shot_samples, compute_few_shot_prompt
    few_shot_samples = get_few_shot_samples(SOURCE_FILE, num_samples=FEW_SHOT_NUM)
    few_shot_samples = [compute_few_shot_prompt(item, prompt=FEW_SHOT_PROMPT['template']) for item in few_shot_samples]
    few_shot_prompt = f'{FEW_SHOT_PROMPT["prefix"]}\n'
    few_shot_prompt += '\n'.join(few_shot_samples)
    few_shot_prompt += f'\n{FEW_SHOT_PROMPT["suffix"]}\n'

# %%
prompt_prefix = PREDICT_PROMPT

# %%
all_examples = []
ask_list = []

error_type = []
for error_item in error_type_item:
    error_type.append(error_item['name'])
error_type_content = json.dumps(error_type, ensure_ascii=False)

for idx, response_item in tqdm(enumerate(ori_data)):
    id = response_item['id']
    question = response_item.get('question', '')
    reference = response_item.get('reference', '')
    analysis = response_item.get('analysis', '')
    total = response_item.get('total', '')
    manual_label = response_item.get('manual_label', '')
    steps = response_item.get('steps', '')
    cache_content = response_item['cache'] if 'cache' in response_item else False

    reponse_content = []
    for s_idx, step in enumerate(steps):
        response = step['response']
        reponse_content.append(f'## Step {s_idx}. {response}')
    
    # Construct Q&A session content (context + questions + references)
    ask_content = prompt_prefix.format(
        question=question,
        total=total,
        score_guideline=score_guideline,
        few_shot_samples=few_shot_prompt,
        error_type=error_type_content,
        reference=reference,
        analysis=analysis,
        student_answer=''.join(reponse_content)
    )
    ask_list.append((ask_content, id, cache_content))

save_content = None
save_content_id_dict = {}
def refresh_save_content(id, content):
    global save_content
    global save_content_id_dict
    if save_content is None:
        save_content = []
        for idx, item in enumerate(ask_list):
            save_content.append((item[1], item[2]))
            save_content_id_dict[item[1]] = idx
    idx = save_content_id_dict[id]
    save_content[idx] = (id, content)
    with open(SAVE_FILE, 'w', encoding='utf-8') as f:
        for item in save_content:
            f.write(item[0] + '\t' + json.dumps(item[1], ensure_ascii=False) + '\n')

def build_chat_custom(content):
    content = f'<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
    return content

if args.skip_thinking == '1':
    for idx, tp in enumerate(ask_list):
        ask_list[idx] = (build_chat_custom(tp[0]), tp[1], tp[2])
    
#%%
# Calculate total number of evaluation batches
if llm_name not in API_MODELS:
    num_batches = len(ask_list) // BATCH_SIZE + (1 if len(ask_list) % BATCH_SIZE != 0 else 0)

    # Run batch prediction and persist results (with progress tracking)
    for i in tqdm(range(num_batches)):
        batch = ask_list[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        prompts = [item[0] for item in batch]
        ids = [item[1] for item in batch]
        cache_content_list = [item[2] for item in batch]
        max_length = [len(item[0]) for item in batch]
        max_length.sort(reverse=True)
        max_new_tokens = max_length[0]
        filter_prompts = []
        filter_idxes = []
        output_list = []
        for idx, tp in enumerate(zip(prompts, cache_content_list)):
            prompt, cache = tp
            if cache == False:
                filter_prompts.append(prompt)
                filter_idxes.append(idx)
                output_list.append('')
            else:
                output_list.append(cache)
        
        if len(filter_prompts) > 0:
            outputs = pred(filter_prompts, max_new_tokens=MAX_NEW_TOKENS, build_message=args.skip_thinking != '1', do_sample=args.do_sample == '1', temperature=float(args.temperature), top_p=float(args.top_p))
        else:
            outputs = []
        
        for res, idx in zip(outputs, filter_idxes):
            res = res.replace('\n', '')
            res = res.replace(' ', '')
            output_list[idx] = res
        
        for res, id in zip(output_list, ids):
            refresh_save_content(id, res)
else:
    for ask_content, id, cache in tqdm(ask_list):
        if cache != False:
            res = cache
        else:
            res = pred(ask_content, model=llm_name)
            res = res[0]
            res = res.replace('\n', '')
            res = res.replace(' ', '')
            if res.find("{'total'") >= 0:
                res = res.replace('\'', '"')
        refresh_save_content(id, res)

#%%
