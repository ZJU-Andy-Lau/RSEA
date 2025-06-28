export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -u pretrain_swt.py --dataset_path datasets/ant \
 --encoder_path weights/pretrain_swt_cnn_r2_0409_large \
 --encoder_output_path weights/encoder_ant_0627 \
 --decoder_output_path ./weights/decoders_ant_0627 \
 --max_epoch 300 \
 --batch_size 2 \
 --dataset_num 10 \
