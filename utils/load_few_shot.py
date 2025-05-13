# %%
import json
import random
from tqdm import tqdm

def get_few_shot_samples(file_name, num_samples=3):
    with open(file_name) as f:
        ori_data = f.readlines()
    ori_data = [json.loads(item) for item in ori_data]
    low = []
    mid = []
    high = []
    for item in tqdm(ori_data):
        total = float(item['total'])
        manual_score = float(item['manual_label'])
        norm_score = manual_score / total
        if norm_score < 0.333:
            low.append(item)
        elif norm_score < 0.667:
            mid.append(item)
        else:
            high.append(item)
    results = []
    for i in range(num_samples):
        if i % num_samples == 0 and len(low) > 0:
            results.append(random.choice(low))
            continue
        if i % num_samples == 1 and len(mid) > 0:
            results.append(random.choice(mid))
            continue
        if i % num_samples == 2 and len(high) > 0:
            results.append(random.choice(high))
    return results

def compute_few_shot_prompt(sample, prompt):
    output = {}
    output_steps = []
    steps = sample['steps']
    for s in steps:
        os = {}
        os['step_score'] = int(s['label'])
        os['errors'] = s['errors']
        output_steps.append(os)
    reponse_content = []
    for s_idx, step in enumerate(steps):
        response = step['response']
        reponse_content.append(f'## Step {s_idx}. {response}')
    output['total'] = sample['total']
    output['pred_score'] = sample['manual_label']
    output['steps'] = output_steps
    format_prompt = prompt.format(question=sample['question'], total=sample['total'], reference=sample['reference'], analysis=sample['analysis'], student_answer=''.join(reponse_content), output=json.dumps(output, ensure_ascii=False))
    return format_prompt

# %%
