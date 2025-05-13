from transformers import AutoModelForCausalLM, AutoConfig
from transformers import AutoTokenizer
from peft import LoraConfig, TaskType, PeftModel, PeftModelForCausalLM
from typing import Optional, List
from copy import deepcopy
from transformers.generation import GenerationMixin
import torch

class Predictor(GenerationMixin):
    true_model: PeftModelForCausalLM

    def __init__(self,
                 num_gpus: list = [0],
                 model_from_pretrained: str = None,
                 resume_path: str = None
                 ):
        self.config = AutoConfig.from_pretrained(model_from_pretrained, trust_remote_code=True)
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            target_modules=["c_attn", "c_proj", "w1", "w2"],
            lora_alpha=32,
            lora_dropout=0.1,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_from_pretrained, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
                model_from_pretrained, device_map="auto", trust_remote_code=True).eval()
        self.llm = self.model
        self.generation_config = self.llm.generation_config
        self.model = PeftModel.from_pretrained(
            self.model, resume_path, config=peft_config)

    def predict(self, text='', max_length=150, temperature=1.0):
        with torch.no_grad():
            inputs = self.tokenizer.encode(text)
            input_ids = torch.LongTensor([inputs]).to(self.device)
            output = self.true_model.generate(**{
                'input_ids': input_ids,
                'max_length': max_length,
                'do_sample': False,
                'temperature': temperature
            })
            out_text = self.tokenizer.decode(
                output[0], skip_special_tokens=True)
        return out_text

    @torch.inference_mode()
    def chat(self,
             query: str,
             history = None,
             system: str = "You are a helpful assistant.",
             stop_words_ids: Optional[List[List[int]]] = None,
             generation_config=None,
             **kwargs,):
        tokenizer = self.tokenizer
        generation_config = generation_config if generation_config is not None else self.generation_config

        return self.model.chat(
            tokenizer=tokenizer,
            query=query,
            history=history,
            system=system,
            generation_config=generation_config,
            stop_words_ids=stop_words_ids,
            **kwargs,
        )
    
    def __call__(self, text='', max_length=150, temperature=0):
        return self.predict(text=text, max_length=max_length, temperature=temperature)
