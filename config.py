import torch
from dataclasses import dataclass
from transformers import GenerationConfig


@dataclass
class Configs:
    generation_config = {
        'max_new_tokens': 1024,
        'temperature': 0.2,
        'top_p': 0.95,
        'repetition_penalty': 1.15,
        'do_sample': True
    }
    inference_cfg = {
        'max_length': 1024,
        'tok_batch_size': 16,
        'inference_batch_size': 10,
    }
    
    model_cfg = {
        'pad_to_multiple_of': 8,
    }

    prompt_config = {
        'system_format': "<|system|>\n{system}",
        'system_no_input_prompt': "Below is a query related to banking compliance. Please respond in a formal language",
        'turn_no_input_format': "\n<|user|>\n{instruction}\n<|assistant|>\n"
    }

    bnb_config = dict(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
    )
