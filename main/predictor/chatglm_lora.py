from transformers import AutoModel
from transformers import AutoTokenizer
from peft import LoraConfig, TaskType, PeftModel, PeftModelForCausalLM
from typing import Tuple, List
import json
import torch


class Predictor():
    true_model: PeftModelForCausalLM

    def __init__(self,
                 num_gpus: list = [0],
                 model_from_pretrained: str = None,
                 resume_path: str = None
                 ):
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_from_pretrained, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_from_pretrained, trust_remote_code=True).half().cuda()
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
    
    def build_chat_input(self, query, history=None, role="user"):
        if history is None:
            history = []
        input_ids = []
        for item in history:
            content = item["content"]
            if item["role"] == "system" and "tools" in item:
                content = content + "\n" + json.dumps(item["tools"], indent=4, ensure_ascii=False)
            input_ids.extend(self.tokenizer.build_single_message(item["role"], item.get("metadata", ""), content))
        input_ids.extend(self.tokenizer.build_single_message(role, "", query))
        input_ids.extend([self.tokenizer.get_command("<|assistant|>")])
        return input_ids

    def predict(self, query: str | list = '', history: List = None, max_new_tokens=512, num_beams:int=1, top_p: float = 0.8, temperature=1.0, do_sample: bool = False, build_message=False):
        if isinstance(query, str):
            query = [query]
            history = [history] if history is not None else None
        with torch.no_grad():
            if build_message:
                inputs = []
                batch_max_len = 0
                for i, t in enumerate(query):
                    if history is not None and len(history) > 0:
                        h_unit = history[i]
                        t = self.build_chat_input(t, h_unit)
                    else:
                        t = self.tokenizer.build_single_message("user", "", t)
                        t.extend([self.tokenizer.get_command("<|assistant|>")])
                    if batch_max_len < len(t):
                        batch_max_len = len(t)
                    inputs.append(t)
                for idx, t in enumerate(inputs):
                    remain = batch_max_len - len(t)
                    inputs[idx] = [self.tokenizer.pad_token_id] * remain + t
            else:
                inputs = self.tokenizer(
                        query, padding=True, truncation=True)['input_ids']
            input_ids = torch.LongTensor(inputs).to(self.device)
            output = self.true_model.generate(**{
                'input_ids': input_ids,
                'max_new_tokens': max_new_tokens,
                'num_beams': num_beams,
                'do_sample': do_sample,
                'top_p': top_p,
                "temperature": temperature,
                'do_sample': False
            })
            out_text = self.tokenizer.batch_decode(
                output, skip_special_tokens=True)
            if build_message:
                out_text = [self.true_model.process_response(t, [])[0] for t in out_text]
        return out_text

    @torch.inference_mode()
    def chat(self, query: str, history: List[Tuple[str, str]] = None, role: str = "user",
             max_length: int = 8192, num_beams=1, do_sample=True, top_p=0.8, temperature=0.8, logits_processor=None,
             **kwargs):
        response, history = self.true_model.chat(
            self.tokenizer, query, history, role, max_length, num_beams, do_sample, top_p, temperature, logits_processor, **kwargs)
        return response, history

    @torch.inference_mode()
    def stream_chat(self, query: str, history: List[Tuple[str, str]] = None, role: str = "user",
                    past_key_values=None, max_length: int = 8192, do_sample=True, top_p=0.8, temperature=0.8,
                    logits_processor=None, return_past_key_values=False, **kwargs):
        for result in self.true_model.stream_chat(self.tokenizer, query, history, role, past_key_values, max_length, do_sample, top_p, temperature, logits_processor, return_past_key_values, **kwargs):
            yield result

    def __call__(self, query: str | list = '', history: List = None, max_new_tokens=512, num_beams:int=1, top_p: float = 0.8, temperature=1.0, do_sample: bool = False, build_message=False):
        return self.predict(query=query, history=history, max_new_tokens=max_new_tokens, num_beams=num_beams, top_p=top_p, temperature=temperature, do_sample=do_sample, build_message=build_message)
