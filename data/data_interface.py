import inspect
import importlib
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import random
import torch
import argparse
from transformers import AutoTokenizer
import os
class TrainCollater:
    def __init__(self, prompt_list=None, llm_tokenizer=None, train=False, terminator="\n", max_step=1):
        self.prompt_list = prompt_list
        self.llm_tokenizer = llm_tokenizer
        self.train = train
        self.terminator = terminator
        self.max_step = max_step
        self.cur_step = 1

    def __call__(self, batch):
        # Ensure batch is valid
        if not batch or not isinstance(batch, list) or not batch[0]:
            raise ValueError(f"Invalid batch: {batch}")

        # Prompt construction
        if isinstance(self.prompt_list, list) and self.prompt_list:
            instruction = random.choice(self.prompt_list)
            inputs_text = instruction if isinstance(instruction, list) else [instruction] * len(batch)
        else:
            instruction = batch[0].get("instruction_input", None)
            if instruction is None:
                raise ValueError("No valid instruction_input in batch and prompt_list is unavailable")
            inputs_text = instruction if isinstance(instruction, list) else [instruction] * len(batch)

        # Validate inputs_text immediately
        if not inputs_text or not all(isinstance(text, str) and text.strip() for text in inputs_text):
            raise ValueError(f"Invalid inputs_text after initial setup: {inputs_text}")
        #print(f"Batch size: {len(batch)}")
        #print(f"Inputs text before processing (sample 0): {inputs_text[0]}")

        # Remove flag logic since we want consistent tokens
        p = random.random()
        thresh_hold = 1 - (self.cur_step / self.max_step)
        flag = p <= thresh_hold and self.train  # Still computed but not used for token choice
        if self.train:
            self.cur_step += 1

        # Debug raw batch data
        print(f"{'Training' if self.train else 'Validation'} Batch:")
        print(f"Raw batch sample: {batch[0]}")

        for i, sample in enumerate(batch):
            input_text = inputs_text[i]
            if '[HistoryHere]' in input_text:
                insert_prompt = ", ".join([seq_title + ' [HistoryEmb]' for seq_title in sample['seq_name']])
                input_text = input_text.replace('[HistoryHere]', insert_prompt)
            if '[CansHere]' in input_text:
                insert_prompt = ", ".join([can_title + ' [CansEmb]' for can_title in sample['cans_name']])
                input_text = input_text.replace('[CansHere]', insert_prompt)
            inputs_text[i] = input_text + " Answer with only the Game title: "

        targets_text = [sample['correct_answer'] for sample in batch]

        # Debug processed inputs and targets
        #print(f"Processed prompt (sample 0): {inputs_text[0]}")
        #print(f"Target (sample 0): {targets_text[0]}")

        if self.train:
                    try:
                        targets_text = [target_text.strip() + self.terminator for target_text in targets_text]
                        prompt_tokens = self.llm_tokenizer(inputs_text, return_tensors="pt", padding="longest", truncation=True, max_length=512, add_special_tokens=True, return_attention_mask=True)
                        if prompt_tokens is None or 'input_ids' not in prompt_tokens:
                            raise ValueError("Prompt tokenization failed")
                        target_tokens = self.llm_tokenizer(targets_text, return_tensors="pt", padding="longest", truncation=True, max_length=512, add_special_tokens=False, return_attention_mask=True)
                        if target_tokens is None or 'input_ids' not in target_tokens:
                            raise ValueError("Target tokenization failed")
                        batch_tokens = {
                            "input_ids": torch.cat([prompt_tokens["input_ids"], target_tokens["input_ids"]], dim=1),
                            "attention_mask": torch.cat([prompt_tokens["attention_mask"], target_tokens["attention_mask"]], dim=1)
                        }
                        labels = batch_tokens["input_ids"].clone()
                        prompt_length = prompt_tokens["input_ids"].shape[1]
                        for i in range(len(batch)):
                            labels[i, :prompt_length] = -100
                        # Debug
                        history_emb_id = self.llm_tokenizer.convert_tokens_to_ids('[HistoryEmb]')
                        cans_emb_id = self.llm_tokenizer.convert_tokens_to_ids('[CansEmb]')
                       # print(f"Training Batch Debug:")
                       # print(f"Expected IDs - [HistoryEmb]: {history_emb_id}, [CansEmb]: {cans_emb_id}")
                       # print(f"Prompt tokens (sample 0): {prompt_tokens['input_ids'][0]}")
                       # print(f"Contains [HistoryEmb]? {history_emb_id in prompt_tokens['input_ids'][0]}")
                        #print(f"Contains [CansEmb]? {cans_emb_id in prompt_tokens['input_ids'][0]}")
                        #print(f"Target tokens (sample 0): {target_tokens['input_ids'][0]}")
                        #print(f"Combined input IDs (sample 0): {batch_tokens['input_ids'][0]}")
                        #print(f"Labels (sample 0): {labels[0]}")
                        new_batch = {
                            "tokens": batch_tokens,
                            "labels": labels,
                            "seq": torch.stack([torch.tensor(sample['seq'], dtype=torch.long) for sample in batch], dim=0),
                            "cans": torch.stack([torch.tensor(sample['cans'], dtype=torch.long) for sample in batch], dim=0),
                            "len_seq": torch.stack([torch.tensor(sample['len_seq'], dtype=torch.long) for sample in batch], dim=0),
                            "len_cans": torch.stack([torch.tensor(sample['len_cans'], dtype=torch.long) for sample in batch], dim=0),
                            "item_id": torch.stack([torch.tensor(sample['item_id'], dtype=torch.long) for sample in batch], dim=0),
                            "flag": flag,
                            "correct_answer": targets_text,
                            "cans_name": [sample['cans_name'] for sample in batch]
                        }
                        return new_batch  # Return inside try
                    except Exception as e:
                        print(f"Error in TrainCollater (train): {str(e)}")
                        print(f"Inputs text (sample 0): {inputs_text[0] if inputs_text else 'None'}")
                        print(f"Targets text (sample 0): {targets_text[0] if targets_text else 'None'}")
                        raise
        else:
                    try:
                        if not inputs_text:
                            raise ValueError("inputs_text is empty or None")
                        if not all(isinstance(text, str) for text in inputs_text):
                            raise ValueError(f"inputs_text contains non-string elements: {inputs_text}")
                        if not all(text.strip() for text in inputs_text):
                            raise ValueError(f"inputs_text contains empty strings after stripping: {inputs_text}")
                        print(f"Validation/Test Inputs (sample 0): {inputs_text[0]}")

                        batch_tokens = self.llm_tokenizer(inputs_text, return_tensors="pt", padding="longest", truncation=True, max_length=512, add_special_tokens=True, return_attention_mask=True)
                        if batch_tokens is None or 'input_ids' not in batch_tokens:
                            raise ValueError("Tokenization failed in validation")
                        
                        cans_name = [sample['cans_name'] for sample in batch]
                        history_emb_id = self.llm_tokenizer.convert_tokens_to_ids('[HistoryEmb]')
                        cans_emb_id = self.llm_tokenizer.convert_tokens_to_ids('[CansEmb]')
                        if history_emb_id is None or cans_emb_id is None:
                            raise ValueError(f"Special token IDs not found: [HistoryEmb]={history_emb_id}, [CansEmb]={cans_emb_id}")
                       # print(f"Validation/Test Batch Debug:")
                       # print(f"Batch sample (0): {batch[0]}")
                       # print(f"Expected IDs - [HistoryEmb]: {history_emb_id}, [CansEmb]: {cans_emb_id}")
                       # print(f"Batch tokens (sample 0): {batch_tokens['input_ids'][0]}")
                       # print(f"Contains [HistoryEmb]? {history_emb_id in batch_tokens['input_ids'][0]}")
                       # print(f"Contains [CansEmb]? {cans_emb_id in batch_tokens['input_ids'][0]}")

                        new_batch = {
                            "tokens": batch_tokens,
                            "seq": torch.stack([torch.tensor(sample['seq'], dtype=torch.long) for sample in batch], dim=0),
                            "cans": torch.stack([torch.tensor(sample['cans'], dtype=torch.long) for sample in batch], dim=0),
                            "len_seq": torch.stack([torch.tensor(sample['len_seq'], dtype=torch.long) for sample in batch], dim=0),
                            "len_cans": torch.stack([torch.tensor(sample['len_cans'], dtype=torch.long) for sample in batch], dim=0),
                            "item_id": torch.stack([torch.tensor(sample['item_id'], dtype=torch.long) for sample in batch], dim=0),
                            "flag": flag,
                            "correct_answer": targets_text,
                            "cans_name": cans_name
                        }
                        return new_batch  # Return inside try
                    except Exception as e:
                        print(f"Error in TrainCollater (val/test): {str(e)}")
                        print(f"Batch: {batch if batch else 'None'}")
                        print(f"Inputs text: {inputs_text if inputs_text else 'None'}")
                        raise

        return new_batch

class DInterface(pl.LightningDataModule):
    def __init__(self, llm_tokenizer=None, num_workers=8, dataset='', **kwargs):
        super().__init__()
        self.num_workers = num_workers
        self.llm_tokenizer = llm_tokenizer
        self.dataset = dataset
        self.kwargs = kwargs
        self.batch_size = kwargs['batch_size']
        self.max_epochs = kwargs['max_epochs']
        self.load_data_module()
        self.load_prompt(kwargs['prompt_path'])

        self.trainset = self.instancialize(stage='train')
        self.valset = self.instancialize(stage='val')
        self.testset = self.instancialize(stage='test')
        self.max_steps = self.max_epochs * (len(self.trainset) // self.batch_size) // self.num_workers

    def train_dataloader(self):
        return DataLoader(self.trainset,
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers, 
                          shuffle=True,
                          drop_last=True,
                          collate_fn=TrainCollater(prompt_list=self.prompt_list, llm_tokenizer=self.llm_tokenizer, train=True, max_step=self.max_steps))

    def val_dataloader(self):
        return DataLoader(self.valset, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers, 
                          shuffle=False,
                          collate_fn=TrainCollater(prompt_list=self.prompt_list, llm_tokenizer=self.llm_tokenizer, train=False))

    def test_dataloader(self):
        return DataLoader(self.testset, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers, 
                          shuffle=False,
                          collate_fn=TrainCollater(prompt_list=self.prompt_list, llm_tokenizer=self.llm_tokenizer, train=False))

    def load_data_module(self):
        name = self.dataset
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            self.data_module = getattr(importlib.import_module('.' + name, package=__package__), camel_name)
        except:
            raise ValueError(f'Invalid Dataset File Name or Invalid Class Name data.{name}.{camel_name}')

    def instancialize(self, **other_args):
        class_args = inspect.getfullargspec(self.data_module.__init__).args[1:]
        inkeys = self.kwargs.keys()
        args1 = {arg: self.kwargs[arg] for arg in class_args if arg in inkeys}
        args1.update(other_args)
        return self.data_module(**args1)
    
    def load_prompt(self, prompt_path):
        self.prompt_list = []
        if os.path.isdir(prompt_path):
            print(f"Loading prompts from directory: {prompt_path}")
            for filename in os.listdir(prompt_path):
                if filename.endswith(".txt"):
                    filepath = os.path.join(prompt_path, filename)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        self.prompt_list.extend([line.strip() for line in f.readlines()])
        elif os.path.isfile(prompt_path):
            print(f"Loading prompts from file: {prompt_path}")
            filepath = prompt_path  # Use prompt_path directly as filepath
            with open(filepath, 'r', encoding='utf-8') as f:
                self.prompt_list = [line.strip() for line in f.readlines()]
        else:
            raise ValueError(f"Prompt path {prompt_path} is neither a file nor a directory!")
        
        if not self.prompt_list:
            raise ValueError(f"No prompts loaded from {prompt_path}!")
        
        valid_prompts = [p for p in self.prompt_list if '[HistoryHere]' in p and '[CansHere]' in p]
        if not valid_prompts:
            raise ValueError("No prompts contain required placeholders [HistoryHere] and [CansHere]!")
        
        self.prompt_list = valid_prompts
        print(f"Loaded {len(self.prompt_list)} valid prompts")