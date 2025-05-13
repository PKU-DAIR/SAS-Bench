import os
import json
import torch
import random
import pickle
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import PretrainedConfig, PreTrainedTokenizer

class LLMPureTextDataset(Dataset):

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
    
    def __getitem__(self, index):
        index = self.random_list[index]
        data = self.data[index]
        context = data['context']
        target = data['target']

        mx = self.max_length // 2
        context_ids = self.tokenizer.encode(context, max_length=mx, truncation=True)
        target_ids = self.tokenizer.encode(
            target,
            max_length=mx,
            truncation=True,
            add_special_tokens=False)
        ids = context_ids + target_ids + [self.config.eos_token_id]
        input_len = len(ids)
        context_len = len(context_ids)
        labels = [-100] * (context_len - 1) + ids[context_len - 1:] + [-100] * (self.max_length - input_len)

        f_ids = ids + [self.config.pad_token_id] * (self.max_length - input_len)

        input_ids = torch.tensor(f_ids)
        labels = torch.tensor(labels)

        return {
            'input_ids': input_ids,
            'labels': labels
        }
    
    def __len__(self):
        return len(self.data)