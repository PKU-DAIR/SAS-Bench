import os
import datetime
import numpy as np


class Analysis():

    def __init__(self):
        self.train_record = {}
        self.eval_record = {}
        self.model_record = {}

    '''
    append data record of train
    train_record_item: dict
    '''

    def append_train_record(self, train_record_item):
        for key in train_record_item:
            if key not in self.train_record:
                self.train_record[key] = []
            self.train_record[key].append(train_record_item[key])

    '''
    append data record of eval
    eval_record_item: dict
    '''

    def append_eval_record(self, eval_record_item):
        for key in eval_record_item:
            if key not in self.eval_record:
                self.eval_record[key] = []
            self.eval_record[key].append(eval_record_item[key])

    '''
    append data record of model
    uid: model uid
    '''

    def append_model_record(self, uid):
        key = "model_uid"
        if key not in self.model_record:
            self.model_record[key] = []
        self.model_record[key].append(uid)

    def save_all_records(self, uid):
        self.save_record('train_record', uid)
        self.save_record('eval_record', uid)
        self.save_record('model_record', uid)

    def save_record(self, record_name, uid):
        record_dict = getattr(self, record_name)
        path = f'./data_record/{uid}'
        if not os.path.exists(path):
            os.makedirs(path)
        head = []
        for key in record_dict:
            head.append(key)
        if len(head) == 0:
            return uid
        result = ''
        for idx in range(len(record_dict[head[0]])):
            for key in head:
                result += str(record_dict[key][idx]) + '\t'
            result += '\n'

        result = "\t".join(head) + '\n' + result

        with open(f'{path}/{record_name}.csv', encoding='utf-8', mode='w+') as f:
            f.write(result)

        return uid
