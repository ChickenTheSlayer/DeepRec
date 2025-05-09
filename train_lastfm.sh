python main.py \
--mode train \
--batch_size 8 \
--accumulate_grad_batches 16 \
--dataset lastfm_data \
--data_dir data/ref/lastfm \
--cans_num 20 \
--prompt_path ./prompt/artist.txt \
--rec_embed SASRec \
--llm_tuning lora \
--llm_path xxx \
--rec_model_path ./rec_model/lastfm.pt \
--ckpt_dir ./checkpoints/lastfm/ \
--output_dir ./output/lastfm/ \
--log_dir lastfm_logs \
--lr_warmup_start_lr 7e-6 \
--lr 7e-4 \
--lr_decay_min_lr 7e-6 \
--max_epochs 5