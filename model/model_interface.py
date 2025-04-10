import inspect
import torch
import importlib
from torch import nn
from torch.nn import functional as F
import torch.optim.lr_scheduler as lrs
import pandas as pd
import pytorch_lightning as pl
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import random
import os.path as op
import os
from optims import LinearWarmupCosineLRScheduler
from peft import LoraConfig, TaskType, get_peft_model
import re
from fuzzywuzzy import fuzz

class SingleTitleStopper(StoppingCriteria):
    def __init__(self, tokenizer, title_end_tokens=["\n"]):
        self.tokenizer = tokenizer
        self.end_pattern = [tokenizer.encode(t, add_special_tokens=False)[0] for t in title_end_tokens]
        self.min_title_length = 4  # Ensure full titles

    def __call__(self, input_ids, scores, **kwargs):
        last_token = input_ids[0, -1].item()
        sequence_length = input_ids.shape[-1]
        return last_token in self.end_pattern and sequence_length >= self.min_title_length + 1

class MaxLengthCriteria(StoppingCriteria):
    def __init__(self, max_length):
        self.max_length = max_length

    def __call__(self, input_ids, scores, **kwargs):
        return input_ids.shape[-1] >= self.max_length

class MInterface(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.load_llm(self.hparams.llm_path)
        self.load_rec_model(self.hparams.rec_model_path)
        self.load_projector()
        self.validation_outputs = []
        self.test_outputs = []

        if self.hparams.ckpt_path and os.path.exists(self.hparams.ckpt_path):
            print(f"Loading partial checkpoint from {self.hparams.ckpt_path}")
            ckpt = torch.load(self.hparams.ckpt_path, map_location='cpu')
            missing, unexpected = self.load_state_dict(ckpt['state_dict'], strict=False)
            print(f"Loaded checkpoint. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

    def wrap_emb(self, batch):
        input_embeds = self.llama_model.get_input_embeddings()(batch["tokens"]["input_ids"])
        print(f"Input embeds shape before replacement: {input_embeds.shape}")
        his_token_id = self.llama_tokenizer.convert_tokens_to_ids("[HistoryEmb]")
        cans_token_id = self.llama_tokenizer.convert_tokens_to_ids("[CansEmb]")
        item_token_id = self.llama_tokenizer.convert_tokens_to_ids("[ItemEmb]")
        #print(f"Special token IDs: {{'[HistoryEmb]': {his_token_id}, '[CansEmb]': {cans_token_id}, '[ItemEmb]': {item_token_id}}}")
        #print(f"Full input IDs (sample 0): {batch['tokens']['input_ids'][0]}")  # Fix here too

        his_item_embeds = self.encode_items(batch["seq"])
        cans_item_embeds = self.encode_items(batch["cans"])
        item_embeds = self.encode_items(batch["item_id"])
        #print(f"his_item_embeds shape: {his_item_embeds.shape}")
        #print(f"cans_item_embeds shape: {cans_item_embeds.shape}")
        #print(f"item_embeds shape: {item_embeds.shape}")

        for i in range(len(batch["len_seq"])):
            his_mask = batch["tokens"]["input_ids"][i] == his_token_id  # Fix here
            his_indices = his_mask.nonzero(as_tuple=True)[0]
            valid_his = min(len(his_indices), his_item_embeds.shape[1])
            #print(f"Batch {i}: HistoryEmb indices: {his_indices}")
            if valid_his > 0:
                #print(f"Batch {i}: Replacing {valid_his} HistoryEmb at {his_indices[:valid_his]}")
                input_embeds[i, his_indices[:valid_his]] = his_item_embeds[i, :valid_his]

            cans_mask = batch["tokens"]["input_ids"][i] == cans_token_id  # Fix here
            cans_indices = cans_mask.nonzero(as_tuple=True)[0]
            valid_cans = min(len(cans_indices), cans_item_embeds.shape[1])
            #print(f"Batch {i}: CansEmb indices: {cans_indices}")
            if valid_cans > 0:
                #print(f"Batch {i}: Replacing {valid_cans} CansEmb at {cans_indices[:valid_cans]}")
                input_embeds[i, cans_indices[:valid_cans]] = cans_item_embeds[i, :valid_cans]

            item_mask = batch["tokens"]["input_ids"][i] == item_token_id  # Fix here
            item_indices = item_mask.nonzero(as_tuple=True)[0]
            if len(item_indices) > 0:
                #print(f"Batch {i}: Replacing ItemEmb at {item_indices[0]}")
                input_embeds[i, item_indices[0]] = item_embeds[i, 0]

        #print(f"Input embeds shape after replacement: {input_embeds.shape}")
        return input_embeds

    def generate(self, batch, temperature=0.7, do_sample=True, num_beams=1, 
                max_gen_length=20, min_gen_length=4, repetition_penalty=3.0, 
                length_penalty=1.0, num_return_sequences=1, top_p=0.9):
        input_embeds = self.wrap_emb(batch)
        # ...
        stopping_criteria = StoppingCriteriaList([
            SingleTitleStopper(self.llama_tokenizer),
            MaxLengthCriteria(max_length=max_gen_length)
        ])
        bad_words_ids = [[self.llama_tokenizer.encode("<think>", add_special_tokens=False)[0]],
                        [self.llama_tokenizer.encode("Okay", add_special_tokens=False)[0]]]  # Block "Okay"
        generate_kwargs = {
            "inputs_embeds": input_embeds,
            "attention_mask": batch["tokens"]["attention_mask"],
            "max_new_tokens": max_gen_length,
            "min_new_tokens": min_gen_length,
            "repetition_penalty": repetition_penalty,
            "length_penalty": length_penalty,
            "num_return_sequences": num_return_sequences,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": top_p,
            "pad_token_id": self.llama_tokenizer.eos_token_id,
            "eos_token_id": self.llama_tokenizer.eos_token_id,
            #"stopping_criteria": stopping_criteria,
            #"bad_words_ids": bad_words_ids,
        }
        generate_ids = self.llama_model.generate(**generate_kwargs)
        #print("Generated Token IDs:", generate_ids)
        output_text = self.llama_tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        outputs = [self.clean_output(text, batch["cans_name"][i] if "cans_name" in batch and i < len(batch["cans_name"]) else []) for i, text in enumerate(output_text)]
        print("Decoded Outputs Before Processing:", output_text)
        print("Final Decoded Recommendations:", outputs)
        return outputs

    def clean_output(self, text, candidates):
        text = re.sub(r'\[.*?\]', '', text).strip()  # Remove special tokens
        cues = ["Answer with only the movie title:", "recommend is:", "recommended movie:", "suggestion:", "Recommendation:"]
        for cue in cues:
            if cue in text:
                text = text.split(cue)[-1].strip()
                break
        text = text.split('\n')[0].strip()
        print(f"Raw cleaned text: {text}")
        # No forcing; return what the model generates, cleaned
        return text

    def training_step(self, batch, batch_idx):
        if self.scheduler:
            self.scheduler.step(self.trainer.global_step, self.current_epoch, self.trainer.max_steps)
        out = self(batch)
        loss = out.loss
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size)
        self.log('lr', self.scheduler.optimizer.param_groups[0]['lr'], on_step=True, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size)
        self.log('global_step_num', float(self.trainer.global_step), on_step=True, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size)
        print(f"Step {batch_idx} Loss: {loss.item()}, LR: {self.scheduler.optimizer.param_groups[0]['lr']}")
        if batch_idx % 16 == 0:  # Every accumulation (matches accum=16)
            grad_norm = sum(p.grad.norm() for p in self.parameters() if p.grad is not None) or 0  # Fallback
            print(f"Grad Norm: {grad_norm}")
            preds = torch.argmax(out.logits, dim=-1)
            print(f"Pred (0): {self.llama_tokenizer.decode(preds[0][-10:], skip_special_tokens=True)}")
        return loss

    def forward(self, batch):
        input_embeds = self.wrap_emb(batch)
        outputs = self.llama_model(
            inputs_embeds=input_embeds,
            attention_mask=batch["tokens"]["attention_mask"],  # Fix here
            labels=batch["labels"],
            return_dict=True,
            use_cache=False,
            pad_token_id=self.llama_tokenizer.eos_token_id
        )
        return outputs

    def validation_step(self, batch, batch_idx):
            generate_output = self.generate(batch)
            # Debug validation step
            #print(f"Validation Step {batch_idx}:")
            #print(f"Generated Outputs: {generate_output}")
            #print(f"Correct Answers: {batch['correct_answer']}")
            #print(f"Candidate Names: {batch['cans_name']}")
            for i, generate in enumerate(generate_output):
                real = batch['correct_answer'][i]
                cans = batch['cans_name'][i]
                generate = generate.strip().split("\n")[0]
                self.validation_outputs.append({'generate': generate, 'real': real, 'cans': cans})

    def on_validation_epoch_end(self):
        if self.validation_outputs:
            df = pd.DataFrame(self.validation_outputs)
            os.makedirs(self.hparams.output_dir, exist_ok=True)
            df.to_csv(os.path.join(self.hparams.output_dir, 'valid.csv'), index=False)
            valid_ratio, hr = self.calculate_hr1(self.validation_outputs)
            metric = hr * valid_ratio
            self.log('val_prediction_valid', valid_ratio, on_epoch=True, prog_bar=True)
            self.log('val_hr', hr, on_epoch=True, prog_bar=True)
            self.log('metric', metric, on_epoch=True, prog_bar=True)
        self.validation_outputs.clear()

    def test_step(self, batch, batch_idx):
        generate_output = self.generate(batch)
        for i, generate in enumerate(generate_output):
            real = batch['correct_answer'][i]
            cans = batch['cans_name'][i]
            generate = generate.strip().split("\n")[0]
            self.test_outputs.append({'generate': generate, 'real': real, 'cans': cans})

    def on_test_epoch_end(self):
        if self.test_outputs:
            df = pd.DataFrame(self.test_outputs)
            os.makedirs(self.hparams.output_dir, exist_ok=True)
            df.to_csv(os.path.join(self.hparams.output_dir, 'test.csv'), index=False)
            valid_ratio, hr = self.calculate_hr1(self.test_outputs)
            metric = hr * valid_ratio
            self.log('test_prediction_valid', valid_ratio, on_epoch=True, prog_bar=True)
            self.log('test_hr', hr, on_epoch=True, prog_bar=True)
            self.log('metric', metric, on_epoch=True, prog_bar=True)
        self.test_outputs.clear()

    def calculate_hr1(self, eval_content):
        correct_num = 0
        valid_num = 0
        total_num = len(eval_content)
        for item in eval_content:
            generate = item['generate'].strip().lower()
            real = item['real'].strip().lower()
            cans = [c.strip().lower() for c in item['cans']]
            gen_cans_list = [cans_item for cans_item in cans if cans_item in generate]
            if len(gen_cans_list) == 1:
                valid_num += 1
                if real == gen_cans_list[0]:
                    correct_num += 1
        valid_ratio = valid_num / total_num if total_num > 0 else 0
        hr1 = correct_num / valid_num if valid_num > 0 else 0
        return valid_ratio, hr1

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        if self.hparams.lr_scheduler == 'cosine':
            steps_per_epoch = 9574  # Hardcoded from LastFM logs—matches 9,594 steps/epoch
            max_steps = self.hparams.max_epochs * steps_per_epoch // self.hparams.accumulate_grad_batches  # 5 * 9574 // 16 = 2,991
            warmup_steps = 1000
            print(f"Configuring scheduler: steps_per_epoch={steps_per_epoch}, max_steps={max_steps}, warmup_steps={warmup_steps}")
            self.scheduler = LinearWarmupCosineLRScheduler(
                optimizer,
                max_step=max_steps,
                min_lr=self.hparams.lr_decay_min_lr,  # 7e-6
                init_lr=self.hparams.lr,              # 1e-3
                warmup_steps=warmup_steps,
                warmup_start_lr=self.hparams.lr_warmup_start_lr  # 7e-6
            )
            return {
                'optimizer': optimizer,
                'gradient_clip_val': 1.0,
                'gradient_clip_algorithm': 'norm'
            }

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        scheduler.step(self.trainer.global_step, self.current_epoch, self.trainer.max_steps)

    def on_save_checkpoint(self, checkpoint):
        if self.hparams.save == 'part':
            checkpoint.pop('optimizer_states', None)
            to_be_removed = [key for key, value in checkpoint['state_dict'].items() if not self.get_parameter(key).requires_grad]
            for key in to_be_removed:
                checkpoint['state_dict'].pop(key)

    def load_llm(self, llm_path):
        print('Loading DeepSeek model...')
        # Use the tokenizer passed via args (if provided), else load new
        if not hasattr(self, 'llama_tokenizer') or self.llama_tokenizer is None:
            self.llama_tokenizer = AutoTokenizer.from_pretrained(llm_path, use_fast=True, trust_remote_code=True)
        if self.llama_tokenizer.pad_token is None:
            self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        self.llama_tokenizer.padding_side = "right"

        special_tokens = ['[HistoryEmb]', '[CansEmb]', '[ItemEmb]']
        special_tokens_dict = {'additional_special_tokens': special_tokens}
        num_added = self.llama_tokenizer.add_special_tokens(special_tokens_dict)
        #print(f"Added {num_added} special tokens.")
        #print(f"Vocab size before: {len(self.llama_tokenizer) - num_added}, after: {len(self.llama_tokenizer)}")
            #print(f"[HistoryEmb] ID: {self.llama_tokenizer.convert_tokens_to_ids('[HistoryEmb]')}")
        #print(f"[CansEmb] ID: {self.llama_tokenizer.convert_tokens_to_ids('[CansEmb]')}")
        #print(f"Vocab check: {self.llama_tokenizer.get_added_vocab()}")

        test_str = "Test [HistoryEmb] and [CansEmb]"
        test_tokens = self.llama_tokenizer.encode(test_str, add_special_tokens=False)
        #print(f"Test encoding: {test_tokens}")
        #print(f"Contains [HistoryEmb]? {self.llama_tokenizer.convert_tokens_to_ids('[HistoryEmb]') in test_tokens}")
        #print(f"Contains [CansEmb]? {self.llama_tokenizer.convert_tokens_to_ids('[CansEmb]') in test_tokens}")

        self.llama_model = AutoModelForCausalLM.from_pretrained(llm_path, torch_dtype=torch.bfloat16)
        self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))
        #print(f"Model embeddings size: {self.llama_model.get_input_embeddings().weight.shape[0]}")
        # Rest unchanged

        if self.hparams.llm_tuning == 'lora':
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=64,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=['k_proj', 'v_proj', 'q_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
            )
            self.peft_config = peft_config
            self.llama_model = get_peft_model(self.llama_model, peft_config)
            self.llama_model.print_trainable_parameters()
        print('DeepSeek model loading done')

    def load_projector(self):
        name = self.hparams.model_name
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            Model = getattr(importlib.import_module('.' + name, package=__package__), camel_name)
        except:
            raise ValueError(f"Invalid Module File Name or Invalid Class Name {name}.{camel_name}!")
        self.projector = self.instancialize(Model, rec_size=self.hparams.rec_size, llm_size=self.llama_model.config.hidden_size)

    def instancialize(self, Model, **other_args):
        class_args = inspect.getfullargspec(Model.__init__).args[1:]
        inkeys = self.hparams.keys()
        args1 = {arg: getattr(self.hparams, arg) for arg in class_args if arg in inkeys}
        args1.update(other_args)
        return Model(**args1)

    def load_rec_model(self, rec_model_path):
        print('Loading Rec Model')
        self.rec_model = torch.load(rec_model_path, map_location="cpu")
        self.rec_model.eval()
        for name, param in self.rec_model.named_parameters():
            param.requires_grad = False
        print('Loading Rec model Done')

    def encode_items(self, seq):
        if self.hparams.rec_embed == "SASRec":
            item_rec_embs = self.rec_model.cacu_x(seq)
        elif self.hparams.rec_embed in ['Caser', 'GRU']:
            item_rec_embs = self.rec_model.item_embeddings(seq)
        else:
            raise ValueError(f"Unsupported rec_embed: {self.hparams.rec_embed}")
        item_txt_embs = self.projector(item_rec_embs)
        return item_txt_embs