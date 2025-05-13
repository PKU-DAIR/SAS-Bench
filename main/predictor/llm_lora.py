from transformers import AutoModel
from transformers import AutoTokenizer, AutoConfig, LlamaForCausalLM, AutoModelForCausalLM
from peft import LoraConfig, TaskType, PeftModel, PeftModelForCausalLM
from typing import Tuple, List
import json
import torch


class Predictor():
    true_model: PeftModelForCausalLM

    def __init__(self,
                 num_gpus: list = [0],
                 model_from_pretrained: str = None,
                 resume_path: str = None,
                 lora_r=16, lora_alpha=32, lora_dropout=0.1,
                 ):
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self.config = AutoConfig.from_pretrained(
            model_from_pretrained, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_from_pretrained, trust_remote_code=True)
        
        if self.config.model_type == 'chatglm':
            self.model = AutoModel.from_pretrained(
                self.model_from_pretrained, trust_remote_code=True).to(torch.bfloat16)
            self.eos_token_id = self.config.eos_token_id
        elif self.config.model_type == 'llama':
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = LlamaForCausalLM.from_pretrained(
                self.model_from_pretrained, trust_remote_code=True).to(torch.bfloat16)
            terminators = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
            self.eos_token_id = terminators
        elif self.config.model_type == 'qwen':
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_from_pretrained, torch_dtype="auto", device_map="auto", trust_remote_code=True)
        elif self.config.model_type == 'qwen2':
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_from_pretrained, torch_dtype="auto", device_map="auto", trust_remote_code=True)
        
        self.model = PeftModel.from_pretrained(
            self.model, resume_path, config=peft_config)
        self.model_to_device(gpu=num_gpus)

    def model_to_device(self, gpu=[0]):
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.cuda()
        self.model = torch.nn.DataParallel(self.model, device_ids=gpu).cuda()
        self.model.to(self.device)
        self.true_model = self.model.module if hasattr(
            self.model, 'module') else self.model
    
    def process_model_outputs(self, inputs, outputs, tokenizer):
        responses = []
        for input_ids, output_ids in zip(inputs['input_ids'], outputs):
            response = tokenizer.decode(output_ids[len(input_ids):], skip_special_tokens=True).strip()
            responses.append(response)
        return responses
    
    def build_chat_input(self, query:str, history=None):
        if history is None:
            history = []
        history.append(query)
        max_input_tokens = 0
        new_batch_input = self.tokenizer.apply_chat_template(history, add_generation_prompt=True, tokenize=False)
        max_input_tokens = max(max_input_tokens, len(new_batch_input))
        return new_batch_input, max_input_tokens

    def predict(self, query: str | list = '', history: List = None, max_length=512, max_new_tokens=512, num_beams:int=1, top_p: float = 0.8, temperature=1.0, do_sample: bool = False, build_message=False):
        if not isinstance(query, list):
            query = [query]
            history = [history] if history is not None else None
        with torch.no_grad():
            if build_message:
                inputs = []
                batch_max_len = 0
                for i, t in enumerate(query):
                    if isinstance(t, str):
                        t = {'role': 'user', 'content': t}
                    if history is not None and len(history) > 0:
                        h_unit = history[i]
                    else:
                        h_unit = []
                    t, max_input_tokens = self.build_chat_input(t, h_unit)
                    if batch_max_len < max_input_tokens:
                        batch_max_len = max_input_tokens
                    inputs.append(t)
            else:
                inputs = query
                batch_max_len = 0
                for i in range(len(query)):
                    if len(query[i]) > batch_max_len:
                        batch_max_len = len(query[i])
            batched_inputs = self.tokenizer(
                inputs,
                return_tensors="pt",
                padding=True,
                truncation=True).to(self.device)
            if self.config.model_type == 'llama':
                batched_inputs = batched_inputs.data
            batched_outputs = self.true_model.generate(**batched_inputs, **{
                'max_new_tokens': max_new_tokens,
                'num_beams': num_beams,
                'do_sample': do_sample,
                'top_p': top_p,
                "temperature": temperature,
                "eos_token_id": self.eos_token_id
            })
            batched_response = self.process_model_outputs(batched_inputs, batched_outputs, self.tokenizer)
        return batched_response

    def __call__(self, query: str | list = '', history: List = None, max_length=512, max_new_tokens=512, num_beams:int=1, top_p: float = 0.8, temperature=1.0, do_sample: bool = False, build_message=False):
        return self.predict(query, history, max_length, max_new_tokens, num_beams, top_p, temperature, do_sample, build_message)
