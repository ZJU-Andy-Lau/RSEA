#!/bin/bash
#SBATCH --partition=a100x4
#SBATCH --account=wanyi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --job-name=finetune_conv
#SBATCH --output=./log/pretrain0324_log.out



#source ~/.bashrc 
# module load nvidia/cuda/11.6
# module load scl/gcc11.2
# cd $SLURM_SUBMIT_DIR
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -u pretrain_swt.py --dataset_path datasets/WV_1024_ds16 \
 --encoder_path weights/pretrain_swt_cnn_r2_0409_large \
 --encoder_output_path weights/pretrain_swt_cnn_r2_0409_large \
 --decoder_output_path ./weights/ \
 --max_epoch 300 \
 --batch_size 2 \
# --decoder_path weights \
#  --min_loss 10.5 \
#  --decoder_path weights/pretrain_swt_cnn_r2_0409_large \
#4,5,6,7
#pretrain_swt_cnn_r2_0409_large  pretrain_swt_r2_0324_large_b