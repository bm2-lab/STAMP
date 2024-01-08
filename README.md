# STAMP
## Introduction 
This repository hosts the official implementation of STAMP, a method that can predict perturbation outcomes using single-cell RNA-sequencing data from perturbational experimental screens, involving scenarios such as single genetic perturbations, multiple genetic perturbations and perturbation predicion across cell lines.

## Installation
Install [PyG](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html), and then do `python setup.py install`.

## Core API interface
Using this API, you can (1) reproduce the results in our paper and (2) train and test STAMP on your perturbation dataset using a few lines of code.
```python
from stamp import STAMP, load_config
import scanpy as sc

# load config file
config = load_config("./Data/example_config.yml")

# set up and train a STAMP
model = STAMP(config)
model.train()

# load trained model
model.load_pretrained(f"{config['Train']['output_dir']}/trained_models")

# use trained model to predict unseen perturbations
model.prediction(config['Train']['Testing_dataset'], combo_test = True)

# use trained model to predict unseen perturbations; considering Top 40 DEGs
# Top 40 DEGs consisting of Top 20 up-regulation genes and Top 20 down-regulation genes

# load Top 40 test data
top_40_data = sc.read_h5ad("./Data/example_test_top40.h5ad")

# prediction
model.prediction(top_40_data, combo_test = True)
```
## Citation
Yicheng Gao, Zhiting Wei, Qi Liu et al. 
## Contacts
bm2-lab@tongji.edu.cn
