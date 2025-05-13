# %%
import json
from tqdm import tqdm

DATANAME = '7_Math_ShortAns'
IGNORE_ZERO_WITHOUT_REASON = True
NO_IGNORE_REASON = '计算错误'
filename = f'/home/lpc/repos/SAS_Benchmark/backend_data/scores/{DATANAME}.jsonl'
with open(filename) as f:
    ori_data = f.readlines()
ori_data = [json.loads(line) for line in ori_data]

result = []
for item in tqdm(ori_data):
    res_segs = item['bad_student_answer_segs']
    last_idx = 0
    format_response = {
        'id': item['id'],
        'question': item['question'],
        'reference': item['answer'],
        'analysis': item['analysis'],
        'total': item['score'],
        'steps': []
    }
    if 'scoreItem' not in item:
        result.append(format_response)
        continue
    scoreItem = item['scoreItem']
    format_response['manual_label'] = scoreItem['label']
    seg_labels = scoreItem['seg_labels']
    seg_labels = json.loads(seg_labels)
    remain_score = int(float(scoreItem['label']))
    
    for seg_item in seg_labels:
        seg_idx, seg_label, seg_errors = seg_item['idx'], seg_item['label'], seg_item['errors']
        if seg_label == '':
            continue
        if seg_label != '' and int(float(seg_label)) == 0 and len(seg_errors) == 0:
            if IGNORE_ZERO_WITHOUT_REASON:
                continue
            else:
                seg_errors = [NO_IGNORE_REASON]
        if seg_label != '' and int(float(seg_label)) > 0 and len(seg_errors) == 0:
            # if it is english dataset, you may replace it with 'correct'
            seg_errors = ['步骤正确']
        format_response['steps'].append({
            'response': '\n'.join(res_segs[last_idx: seg_idx + 1]),
            'label': seg_label,
            'errors': seg_errors
        })
        last_idx = seg_idx + 1
        if seg_label != '' and int(float(seg_label)) > 0:
            remain_score -= int(float(seg_label))
    if last_idx < len(res_segs):
        format_response['steps'].append({
            'response': '\n'.join(res_segs[last_idx:]),
            'label': 0 if remain_score < 0 else remain_score,
            'errors': []
        })
    result.append(format_response)

with open(f'./datasets/{DATANAME}.jsonl', 'w') as f:
    for item in result:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

# %%
