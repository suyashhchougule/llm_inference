import logging
import os
import time
import torch 
# import fire
import json
import pandas as pd

from typing import Optional
from tqdm import tqdm
from dataclasses import dataclass
from accelerate import Accelerator
from datasets import load_dataset
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding
    )

from dataset import get_loader, process_datasets, process_sft_data

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def get_model_tokenizer(
        model_name: str,
        adapter_checkpoint_dir: str = None,
        bnb_config: Optional[BitsAndBytesConfig] = None, 
        accelerator = None
        ):
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        # padding_side='right',
        truncation=True
        )
    tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.add_special_tokens(
    #     {
    #         "pad_token": "<PAD>",
    #     }
    #     )

    # Load the model
    if bnb_config: 
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_4bit=True,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            device_map='auto'
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map='auto',
            torch_dtype=torch.bfloat16
            )
            
    # Merge the adpater
    if adapter_checkpoint_dir:
        model = PeftModel.from_pretrained(
        model,
        adapter_checkpoint_dir,
        torch_dtype=torch.float16,
        is_trainable= False
        ) 
    # Resize token embeedings to include the pad token
    # model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


def infer(model, tokenizer, instructions):

    model_input = tokenizer(instructions, return_tensors="pt", add_special_tokens=False, padding = True)
    print(model.device)

    generated_ids = model.generate(
        input_ids = model_input['input_ids'].to(model.device),
        attention_mask = model_input['attention_mask'].to(model.device),
        max_new_tokens=500,
        do_sample=True
        )
    
    decoded = tokenizer.batch_decode(generated_ids)
    return decoded

def infer_batch(model, tokenizer, loader, save_results: str = False):
    logger.info(f"Starting batch inference with a batch_size of")
    progress_bar = tqdm(range(len(loader)), total = len(loader))
    for ind, batch in enumerate(loader):
        model_input = tokenizer(batch['text'], return_tensors="pt", add_special_tokens=False, padding = True)
        generated_ids = model.generate(
           input_ids = model_input['input_ids'].to(model.device),
           attention_mask = model_input['attention_mask'].to(model.device),
           max_new_tokens=500, 
           do_sample=True
           )
        decoded = tokenizer.batch_decode(generated_ids)
        for ins, res, annot, ans in zip(batch['Question'], decoded, batch['annotator'], batch['Answer']):
            dic = dict(
                instruction = ins,
                response = res,
                annotator = annot,
                gold_answer = ans
                )
            dic = json.dumps(dic)
            with open(save_results, mode = "a") as f:
                f.write(dic + "\n")
        progress_bar.update(1)
       

if __name__ == "__main__":
    bnb_config = dict(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
    ) 

    # # Load the datasets
    # dataset_paths = [
    #     "/home/atharva/system_evaluation/datasets/nayan.csv",
    #     "/home/atharva/system_evaluation/datasets/tony.csv"
    # ]
    # df = process_datasets(dataset_paths)

    dataset_paths = "dataeaze/sft_dataset"
    df = process_sft_data(dataset_paths)
    loader = get_loader(df, 40)

    # Load the model and tokenizer
    logger.info("Loading model and toknizer")
    model, tokenizer = get_model_tokenizer(
        model_name = "HuggingFaceH4/zephyr-7b-beta",
        # adapter_checkpoint_dir ="/home/atharva/system_evaluation/models/uftm_v1",
        adapter_checkpoint_dir = "/home/atharva/zephyr/train_llm/sftm_v1_exp/checkpoint-12450",
        bnb_config = BitsAndBytesConfig(**bnb_config)
    )
    logger.info("Model and tokenizer loaded successfully")

    # Run the inference
    save_path = "/home/atharva/system_evaluation/output/results_sftm_v1_exp.txt"
    logger.info(f"Saving the results to {os.path.basename(save_path)}")
    infer_batch(model, tokenizer, loader, save_results= save_path) 

