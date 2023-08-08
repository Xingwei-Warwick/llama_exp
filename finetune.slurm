#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=10000
#SBATCH --gres=gpu:ampere_a100:3
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --account=placeholder


module purge
module load GCCcore/11.3.0 CUDA/11.6.0

srun  python sft_trainer.py \
    --model_name 'hf_llama_weights_7B' \
    --dataset_name 'data/formatted_train_data.json' \
    --output_dir './lora_llama_nyt_temporal_graph' \
    --batch_size 32\
    --num_train_epochs 5 \
    --seq_length 4096\
    --use_peft True \
    --peft_lora_r 8 \
    --peft_lora_alpha 16