DATA_PATH=/home/workspace/CSASML/cifar100/
CODE_PATH=/home/workspace/CSASML/ # modify code path here


cd $CODE_PATH && python validate.py $DATA_PATH --model csasml_s18  --checkpoint '/home/data/pyliao/CSASML/output/20240618-093545-pyformer_s18-224/model_best.pth.tar' --dataset 'torch/cifar100' --batch-size 128 --results-file val_result.csv --throughput --latency --output /home/data/pyliao/CSASML/output/20240618-093545-pyformer_s18-224/

