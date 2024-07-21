DATA_PATH=/home/workspace/CASML/cifar100/
CODE_PATH=/home/workspace/CASML/ # modify code path here
CHECKPOINT_PATH=/home/data/pyliao/CASML/output/20240618-093545-casml_s18-224/model_best.pth.tar
OUTPUT_PATH=/home/data/pyliao/CASML/output/20240618-093545-casml_s18-224/

cd $CODE_PATH && python validate.py $DATA_PATH --model casml_s18  --checkpoint $CHECKPOINT_PATH --dataset 'torch/cifar100' --batch-size 128 --results-file val_result.csv --throughput --latency --output $OUTPUT_PATH

