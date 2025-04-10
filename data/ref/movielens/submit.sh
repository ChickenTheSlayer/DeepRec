#!/bin/bash
#SBATCH --partition=M1                    # Partition to use
#SBATCH --qos=q_d8_24                  # QoS assigned to you
#SBATCH --nodes=1                         # Number of nodes (1 in this case)
#SBATCH --gres=gpu:1                      # Request 1 GPU
#SBATCH --time=06:00:00                   # Maximum wall time (adjust as needed, <= 6 hours for your QoS)
#SBATCH --mem=12G                         # Memory allocation (adjust as per QoS limit)
#SBATCH --job-name=train_steam            # Job name
#SBATCH --output=output_train_steam_%j.out  # Output log file
#SBATCH --error=error_train_steam_%j.err   # Error log file

module load anaconda
eval "$(conda shell.bash hook)"
conda activate llara
python modify.py
#python main.py --mode train --batch_size 1 --accumulate_grad_batches 8 --dataset steam_data --data_dir data/ref/steam --cans_num 2 --prompt_path ./prompt/game.txt --rec_embed SASRec --llm_tuning lora --llm_path ./DeepSeek-R1-Distill-Llama-8B --rec_model_path ./rec_model/steam.pt --ckpt_dir ./checkpoints/steam1/ --output_dir ./output/steam/ --log_dir steam_logs --lr_warmup_start_lr 5e-6 --lr 5e-4 --lr_decay_min_lr 5e-6 --max_epochs 1

#python main.py --mode train --batch_size 2 --accumulate_grad_batches 8 --dataset steam_data --data_dir data/ref/steam --cans_num 5 --prompt_path ./prompt/game.txt --rec_embed SASRec --llm_tuning lora --llm_path ./DeepSeek-R1-Distill-Llama-8B --rec_model_path ./rec_model/steam.pt --ckpt_dir ./checkpoints/steam1/ --output_dir ./output/steam/ --log_dir steam_logs --lr_warmup_start_lr 5e-6 --lr 5e-4 --lr_decay_min_lr 5e-6 --max_epochs 5
