import os
import json
import torch
import random
import pickle
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import PretrainedConfig, PreTrainedTokenizer


class ChatGLM_RLHFDataset(Dataset):

    config: PretrainedConfig
    tokenizer: PreTrainedTokenizer

    def __init__(self, tokenizer, config, file_name, max_length=512, do_shuffle=False):
        self.config = config
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.do_shuffle = do_shuffle
        self.data = self.load_jsonl(file_name)
        self.random_list = [idx for idx in range(len(self.data))]
        self.preprocess_list = [self.process_item(item) for item in self.data]
        if self.do_shuffle:
            random.shuffle(self.random_list)

    def load_jsonl(self, file_name):
        with open(file_name, 'r') as f:
            lines = f.readlines()
        data = [json.loads(line) for line in lines]
        return data

    def process_item(self, item):
        conv = item['conversations'] if 'conversations' in item else item

        input_ids, loss_masks = [
            self.tokenizer.get_command('[gMASK]'),
            self.tokenizer.get_command('sop'),
        ], [False, False]

        last_input_len = 0
        last_user_content = ''
        last_assistant_content = ''

        for message in conv:
            if message['role'] in ('system', 'user'):
                loss_mask_val = False
            else:
                loss_mask_val = True

            if message['role'] == 'tool':
                raise NotImplementedError()
            else:
                new_input_ids = self.tokenizer.build_single_message(
                    message['role'], '', message['content']
                )
                new_loss_masks = [loss_mask_val] * len(new_input_ids)
                if message['role'] == 'user':
                    last_user_content = message['content']
                    last_input_len = len(new_input_ids)
                elif message['role'] == 'assistant':
                    last_assistant_content = message['content']
                    role_ids = self.tokenizer.build_single_message(
                        message['role'], '', ''
                    )
                    last_input_len = len(new_input_ids) - len(role_ids)

            input_ids += new_input_ids
            loss_masks += new_loss_masks

        input_ids.append(self.tokenizer.eos_token_id)
        loss_masks = [False, *loss_masks]
        labels = []
        for input_id, mask in zip(input_ids, loss_masks):
            if mask:
                labels.append(input_id)
            else:
                labels.append(-100)
        max_length = self.max_length

        input_ids_without_last_turn = input_ids[:-last_input_len]
        labels_without_last_turn = labels[:-last_input_len]

        exceed_len = len(input_ids) - max_length
        if exceed_len > 0:
            last_input_len = last_input_len - exceed_len
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
        input_ids_without_last_turn = input_ids_without_last_turn[:max_length]
        labels_without_last_turn = labels_without_last_turn[:max_length]
        return {
            'input_ids': input_ids,
            'labels': labels,
            'input_ids_without_last_turn': input_ids_without_last_turn,
            'labels_without_last_turn': labels_without_last_turn,
            'last_input_len': last_input_len,
            'last_user_content': last_user_content,
            'last_assistant_content': last_assistant_content}

    def __getitem__(self, index):
        index = self.random_list[index]
        item = self.data[index]
        input_ids, labels, input_ids_without_last_turn, labels_without_last_turn, last_input_len, last_user_content, last_assistant_content = self.preprocess_list[index].values(
        )
        gold_answers = item['gold_answers']
        bad_answers = item['bad_answers']

        input_ids = torch.tensor(input_ids)
        labels = torch.tensor(labels)
        input_ids_without_last_turn = torch.tensor(input_ids_without_last_turn)
        labels_without_last_turn = torch.tensor(labels_without_last_turn)

        return {
            'query': last_user_content,
            'gold_answers': gold_answers,
            'bad_answers': bad_answers,
            'input_ids': input_ids,
            'input_ids_without_last_turn': input_ids_without_last_turn,
            'labels_without_last_turn': labels_without_last_turn,
            'last_input_len': torch.tensor(last_input_len),
            'last_assistant_content': last_assistant_content,
            'labels': labels
        }

    def __len__(self):
        return len(self.data)
