#!/bin/bash
#SBATCH --partition=M1                    # Partition to use
#SBATCH --qos=q_d8_48                      # QoS assigned to you
#SBATCH --nodes=1                          # Number of nodes (1 in this case)
#SBATCH --gres=gpu:1                       # Request 1 GPU
#SBATCH --time=48:00:00                    # Maximum wall time (adjust as needed, <= 6 hours for your QoS)
#SBATCH --mem=12G                          # Memory allocation (adjust as per QoS limit)
#SBATCH --job-name=train_steam             # Job name
#SBATCH --output=output_train_steam_%j.out # Output log file
#SBATCH --error=error_train_steam_%j.err   # Error log file



module load anaconda
eval "$(conda shell.bash hook)"
conda activate llara

#python test.py

#python main.py --llm_path ./DeepSeek-R1-Distill-Llama-8B --rec_model_path ./rec_model/movielens.pt --ckpt_path ./checkpoints/movielens/last.ckpt --save all --max_epochs 5

#python test.py --llm_path ./DeepSeek-R1-Distill-Llama-8B --rec_model_path ./rec_model/movielens.pt --ckpt_path ./checkpoints/movielens/last.ckpt

#python main.py --mode train --batch_size 1 --accumulate_grad_batches 8 --dataset steam_data --data_dir data/ref/steam --cans_num 2 --prompt_path ./prompt/game.txt --rec_embed SASRec --llm_tuning lora --llm_path ./DeepSeek-R1-Distill-Llama-8B --rec_model_path ./rec_model/steam.pt --ckpt_dir ./checkpoints/steam1/ --output_dir ./output/steam/ --log_dir steam_logs --lr_warmup_start_lr 5e-6 --lr 5e-4 --lr_decay_min_lr 5e-6 --max_epochs 1

#python main.py --mode train --batch_size 3 --accumulate_grad_batches 32 --dataset steam_data --data_dir data/ref/steam --cans_num 20 --prompt_path ./prompt/game.txt --rec_embed SASRec --llm_tuning lora --llm_path ./DeepSeek-R1-Distill-Llama-8B --rec_model_path ./rec_model/steam.pt --ckpt_dir ./checkpoints/steam1/ --output_dir ./output/steam/ --log_dir steam_logs --lr_warmup_start_lr 5e-6 --lr 3e-4 --lr_decay_min_lr 5e-6 --max_epochs 5
python main.py --mode test --batch_size 3 --accumulate_grad_batches 32 --dataset steam_data --data_dir data/ref/steam --cans_num 20 --prompt_path ./prompt/game.txt --rec_embed SASRec --llm_tuning lora --llm_path ./DeepSeek-R1-Distill-Llama-8B --rec_model_path ./rec_model/steam.pt --ckpt_path ./checkpoints/steam1/last-v6.ckpt --output_dir ./output/steam/ --log_dir steam_logs --lr_warmup_start_lr 5e-6 --lr 3e-4 --lr_decay_min_lr 5e-6 --max_epochs 5




#python main.py --mode train --batch_size 5 --accumulate_grad_batches 2 --dataset movielens_data --data_dir data/ref/movielens --cans_num 20 --prompt_path ./prompt/movie.txt --rec_embed SASRec --llm_tuning lora --llm_path ./DeepSeek-R1-Distill-Llama-8B --rec_model_path ./rec_model/movielens.pt --ckpt_dir ./checkpoints/movielens/ --output_dir ./output/movielens/ --log_dir movielens_logs --lr_warmup_start_lr 8e-6 --lr 8e-4 --lr_decay_min_lr 1e-6 --max_epochs 5

#python main.py --mode train --batch_size 2 --accumulate_grad_batches 32 --dataset steam_data --data_dir data/ref/steam --cans_num 20 --prompt_path ./prompt/game.txt --rec_embed SASRec --llm_tuning lora --llm_path ./DeepSeek-R1-Distill-Llama-8B --rec_model_path ./rec_model/steam.pt --ckpt_dir ./checkpoints/steam/ --output_dir ./output/steam/ --log_dir steam_logs --lr_warmup_start_lr 5e-6 --lr 5e-4 --lr_decay_min_lr 5e-6 --max_epochs 5



#python main.py --mode train --batch_size 5 --accumulate_grad_batches 8 --dataset movielens_data --data_dir data/ref/movielens --cans_num 20 --prompt_path ./prompt/movie.txt --rec_embed SASRec --llm_tuning lora --llm_path ./DeepSeek-R1-Distill-Llama-8B --rec_model_path ./rec_model/movielens.pt --ckpt_dir ./checkpoints/movielens/ --output_dir ./output/movielens/ --log_dir movielens_logs --lr_warmup_start_lr 8e-6 --lr 4e-4 --lr_decay_min_lr 1e-6 --max_epochs 5 



#python main.py --mode train --batch_size 5 --accumulate_grad_batches 16 --dataset lastfm_data --data_dir data/ref/lastfm --cans_num 20 --prompt_path ./prompt/artist.txt --rec_embed SASRec --llm_tuning lora --llm_path ./DeepSeek-R1-Distill-Llama-8B --rec_model_path ./rec_model/lastfm.pt --ckpt_dir ./checkpoints/lastfm/ --output_dir ./output/lastfm/ --log_dir lastfm_logs --lr_warmup_start_lr 7e-6 --lr 1e-3 --lr_decay_min_lr 7e-6 --max_epochs 5
#python main.py --mode test --batch_size 5 --accumulate_grad_batches 16 --dataset lastfm_data --data_dir data/ref/lastfm --cans_num 20 --prompt_path ./prompt/artist.txt --rec_embed SASRec --llm_tuning lora --llm_path ./DeepSeek-R1-Distill-Llama-8B --rec_model_path ./rec_model/lastfm.pt --ckpt_dir ./checkpoints/lastfm/last-v1.ckpt --output_dir ./output/lastfm/ --log_dir lastfm_logs --lr_warmup_start_lr 7e-6 --lr 1e-3 --lr_decay_min_lr 7e-6 --max_epochs 5

#python trainingsas.py