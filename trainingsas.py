import torch
from transformers import AutoTokenizer
import argparse
import os
from sasrec import SASRec
from model.model_interface import MInterface

def load_model(llm_path, rec_model_path, ckpt_path):
    args = argparse.Namespace(
        llm_path=llm_path,
        rec_model_path=rec_model_path,
        ckpt_path=ckpt_path,
        llm_tuning='lora',
        peft_dir=None,
        peft_config=None,
        lora_r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        model_name='mlp_projector',
        rec_size=64,
        padding_item_id=1682,
        prompt_path='./prompt/movie.txt',
        data_dir='data/ref/movielens',
        rec_embed='SASRec',
        batch_size=1,
        precision='bf16'
    )
    model = MInterface(**vars(args))
    
    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location='cpu')
        missing, unexpected = model.load_state_dict(ckpt['state_dict'], strict=False)
        print(f"Loaded checkpoint. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
        print(f"Epoch: {ckpt.get('epoch', 0)}, Step: {ckpt.get('global_step', 0)}")
    else:
        raise FileNotFoundError(f"Checkpoint {ckpt_path} not found")

    return model.llama_model, model.llama_tokenizer, model.rec_model, model

def generate_recommendation(llama_model, tokenizer, rec_model, m_interface, watch_history, candidates, device='cpu'):
    prompt = f"This user has watched {'[HistoryEmb], '.join(watch_history)}[HistoryEmb] in the previous. Recommend one movie from: {'[CansEmb], '.join(candidates)}[CansEmb]. Answer:"
    
    batch_tokens = tokenizer(prompt, return_tensors='pt', padding="longest", truncation=True, max_length=512, add_special_tokens=True, return_attention_mask=True)
    
    movie_ids = {"The Matrix": 1, "Inception": 2, "Interstellar": 3, "Toy Story": 4, "Finding Nemo": 5,
                 "Blade Runner": 6, "The Dark Knight": 7, "Dune": 8, "Star Wars": 9}
    seq_ids = [movie_ids.get(movie, 0) for movie in watch_history]
    cans_ids = [movie_ids.get(movie, 0) for movie in candidates]
    
    batch = {
        "tokens": batch_tokens.to(device),
        "seq": torch.tensor([seq_ids], dtype=torch.long).to(device),
        "cans": torch.tensor([cans_ids], dtype=torch.long).to(device),
        "len_seq": [len(watch_history)],
        "len_cans": [len(candidates)],
        "item_id": torch.tensor([[0]], dtype=torch.long).to(device),
        "correct_answer": ["dummy"],
        "cans_name": [candidates]
    }
    
    with torch.autocast(device_type='cuda' if 'cuda' in device else 'cpu', dtype=torch.bfloat16):
        input_embeds = m_interface.wrap_emb(batch)

    with torch.no_grad():
        outputs = llama_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=batch["tokens"]["attention_mask"],
            max_new_tokens=20,
            min_new_tokens=1,
            num_beams=5,
            early_stopping=False,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        #print(f"Generated token IDs: {outputs}")
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    recommendation = generated_text[len(prompt):].strip()
    return recommendation

def main():
    parser = argparse.ArgumentParser(description="Test SASRec recommendation with LoRA checkpoint")
    parser.add_argument('--llm_path', type=str, default='./DeepSeek-R1-Distill-Llama-8B', help='Path to DeepSeek LLaMA model')
    parser.add_argument('--rec_model_path', type=str, default='./rec_model/movielens.pt', help='Path to SASRec model')
    parser.add_argument('--ckpt_path', type=str, default='./checkpoints/movielens/last.ckpt', help='Path to checkpoint')  # Updated to your original
    parser.add_argument('--data_dir', type=str, default='data/ref/movielens', help='Data directory')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    llama_model, tokenizer, rec_model, m_interface = load_model(args.llm_path, args.rec_model_path, args.ckpt_path)
    llama_model.to(device)
    rec_model.to(device)
    m_interface.to(device)

    watch_history = ["The Matrix", "Inception", "Interstellar"]
    candidates = ["Blade Runner", "The Dark Knight", "Dune", "Star Wars"]

    recommendation = generate_recommendation(llama_model, tokenizer, rec_model, m_interface, watch_history, candidates, device)
    print(f"Recommended movie: {recommendation}")

if __name__ == '__main__':
    main()