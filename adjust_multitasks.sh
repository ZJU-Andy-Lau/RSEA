export CUDA_VISIBLE_DEVICES=5
bns=(1 2)
gss=(2000 3000 4000 5000 6000 7000 8000 9000 10000)
for bn in "${bns[@]}"; do
  for gs in "${gss[@]}"; do
  echo "=================================================="
  echo "Running with: Block_Num=${bn}, Grid_Size=${gs}"
  echo "=================================================="
  python -u adjust_zero.py --root datasets/test0615/batch_0 --mapper_blocks_num "$bn" --grid_size "$gs" --log_postfix 0

  done
done