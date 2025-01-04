# export CUDA_VISIBLE_DEVICES=0
# torchrun --standalone --nnodes 1 --nproc-per-node 1 /home/DavidHong/code/git_clone/openvla/vla-scripts/finetune.py \
#   --vla_path "/home/jyr_new/openvla/download_hf/openvla-7b" \
#   --data_root_dir /root/tensorflow_datasets/agile_dataset \
#   --dataset_name agile_dataset \
#   --run_root_dir /home/DavidHong/code/git_clone/openvla/openvla_logs \
#   --adapter_tmp_dir /home/DavidHong/code/git_clone/openvla/openvla_adapter \
#   --lora_rank 32 \
#   --batch_size 16 \
#   --grad_accumulation_steps 1 \
#   --learning_rate 5e-4 \
#   --image_aug True \
#   --wandb_project "openvla" \
#   --wandb_entity "openvla_david" \
#   --save_steps 1000

torchrun --nproc-per-node 1 --master_port=29509 /home/DavidHong/code/git_clone/openvla/vla-scripts/finetune.py \
  --vla_path "/home/jyr_new/openvla/download_hf/openvla-7b" \
  --data_root_dir "/home/DavidHong/data/agile_data_tensorflow_datasets" \
  --dataset_name "agile_dataset" \
  --run_root_dir "/home/DavidHong/code/git_clone/openvla/openvla_logs" \
  --adapter_tmp_dir "/home/DavidHong/code/git_clone/openvla/openvla_adapter" \
  --lora_rank 32 \
  --batch_size 16 \
  --save_steps 500\
  --max_steps 2000 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug False \
  --save_latest_checkpoint_only True