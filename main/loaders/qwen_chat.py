import os
import json
import torch
import random
import pickle
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import PretrainedConfig, PreTrainedTokenizer

class QwenChatDataset(Dataset):

    config: PretrainedConfig
    tokenizer: PreTrainedTokenizer
    
    def __init__(self, tokenizer, config, file_name, max_length=512, do_shuffle=False):
        self.config = config
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.do_shuffle = do_shuffle
        self.data = self.load_jsonl(file_name)
        self.random_list = [idx for idx in range(len(self.data))]
        if self.do_shuffle:
            random.shuffle(self.random_list)
            
    
    def load_jsonl(self, file_name):
        with open(file_name, 'r') as f:
            data = [json.loads(line) for line in f]
        return data
    
    def process_item(self, item):
        conv = item['conversations'] if 'conversations' in item else item

        input_ids, loss_masks = [], []

        # im_start, im_end = "<|im_start|>", "<|im_end|>"
        im_start_tokens = [self.tokenizer.im_start_id]
        im_end_tokens = [self.tokenizer.im_end_id]
        nl_tokens = self.tokenizer.encode("\n")

        def _tokenize_str(role, content):
            return f"{role}\n{content}", self.tokenizer.encode(
                role, allowed_special=set()
            ) + nl_tokens + self.tokenizer.encode(content, allowed_special=set())

        for message in conv:
            if message['role'] in ('system', 'user'):
                loss_mask_val = False
            else:
                loss_mask_val = True

            if message['role'] == 'tool':
                raise NotImplementedError()
            else:
                _, msg_tokens = _tokenize_str(message['role'], message['content'])
                new_input_ids = im_start_tokens + msg_tokens + im_end_tokens
                new_input_ids = new_input_ids + nl_tokens
                new_loss_masks = [loss_mask_val] * len(new_input_ids)

            input_ids += new_input_ids
            loss_masks += new_loss_masks

        input_ids = input_ids + im_end_tokens
        loss_masks = [False, *loss_masks]
        labels = []
        for input_id, mask in zip(input_ids, loss_masks):
            if mask:
                labels.append(input_id)
            else:
                labels.append(-100)
        max_length = self.max_length
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
        return {'input_ids': input_ids, 'labels': labels}
    
    def __getitem__(self, index):
        index = self.random_list[index]
        data = self.data[index]
        input_ids, labels = self.process_item(data).values()

        input_ids = torch.tensor(input_ids)
        labels = torch.tensor(labels)

        return {
            'input_ids': input_ids,
            'labels': labels
        }
    
    def __len__(self):
        return len(self.data)