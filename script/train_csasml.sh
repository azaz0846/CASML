DATA_PATH=/home/workspace/CSASML/cifar100/
CODE_PATH=/home/workspace/CSASML/ # modify code path here


ALL_BATCH_SIZE=64
NUM_GPU=2
GRAD_ACCUM_STEPS=2 # Adjust according to your GPU numbers and memory size.
let BATCH_SIZE=ALL_BATCH_SIZE/NUM_GPU/GRAD_ACCUM_STEPS


cd $CODE_PATH && TORCH_DISTRIBUTED_DEBUG=INFO torchrun --nnodes=1 --nproc_per_node=$NUM_GPU train.py $DATA_PATH \
--dataset 'torch/cifar100' --dataset-download --model csasml_s18 --opt lamb --lr 1e-3 --warmup-epochs 20 \
-b $BATCH_SIZE --grad-accum-steps $GRAD_ACCUM_STEPS \
--drop-path 0.15 --head-dropout 0.0 --epochs 300 --weight-decay 0.001 --input-size 3 224 224  --output /home/data/pyliao/CSASML/output --compute-flops

#torchrun --nnodes=1 --nproc_per_node=$NUM_GPU
