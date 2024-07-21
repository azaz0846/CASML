# CASML (WOCC2024)
A lightweight Transformer model, combined with a split network and a method for extracting cross-scale features for classification tasks, called CASML.

### Architecture
![CASML](https://github.com/azaz0846/CASML/blob/main/figures/CASML.jpg)

### Baseline Compare
![Baseline](https://github.com/azaz0846/CASML/blob/main/figures/baseline_compare.jpg)

## Requirements
- torch>=1.7.0
- torchvision>=0.8.0
- pyyaml
- timm==0.6.11

It is recommended to use Docker to set up the environment.
```sh
docker pull nvcr.io/nvidia/pytorch:23.07-py3
python -m pip install --upgrade pip
pip install -U 'git+https://github.com/facebookresearch/fvcore'
pip install timm==0.6.11
pip install grad-cam
```

## Training
```sh
ch script
bash train_casml.sh
```

## Validation
```sh
ch script
bash val_.sh
```
You need to check the location of the dataset and the output.

### This Token Mixer that uses the attention mechanism trained on CIFAR-100
| Model | Resolution | Params | MACs | Top1 Acc | Download |
| :---     |   :---:    |  :---: |  :---:  |  :---:  |  :---:  |
| CASML | 224 | 23.16 M | 5.9 G |  84.15 | [here](https://github.com/azaz0846/CASML/releases/download/checkpoint-v1/model_best.pth.tar) |
