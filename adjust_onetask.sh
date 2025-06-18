export CUDA_VISIBLE_DEVICES=5
python -u adjust_zero.py --root datasets/test0615/batch_0 --mapper_blocks_num 3 --grid_size 300 --log_postfix 0