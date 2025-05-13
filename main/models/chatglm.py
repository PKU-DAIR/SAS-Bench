from re import A
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class CCGPTModel():

    def __init__(
        self,
        model_name: str = None,
        model_from_pretrained: str = None,
        model_config_file_name: str = None,
        pretrained_file_name: str = None,
    ):
        '''
        CCGPTModel: 生成式模型 (Generative model)

        ### Args:
            `model_name`: 模型名称 (the name of the model)
            `model_from_pretrained`: 从预训练模型中加载模型 (load model from pretrained model)
            `model_config_file_name`: bert配置文件名 (bert config file name)
            `pretrained_file_name`: 预训练模型文件名 (pretrained file name)
            `tagset_size`: 标签数量 (the number of tags)
        '''
        self.model_name = model_name
        self.model_from_pretrained = model_from_pretrained
        self.model_config_file_name = model_config_file_name
        self.pretrained_file_name = pretrained_file_name

        if self.model_name is None:
            raise ValueError("model_name is required")
        if self.model_from_pretrained is None:
            if self.model_config_file_name is None:
                raise ValueError("model_config_file_name is required")
            if self.pretrained_file_name is None:
                raise ValueError("pretrained_file_name is required")

        self.load_model()

    def load_model(self):
        if self.model_name == 'ChatGLM2-6B':
            self.model = AutoModel.from_pretrained(self.model_from_pretrained, trust_remote_code=True).half().cuda()

    def get_model(self):
        return self.model

    def __call__(self):
        return self.get_model()
