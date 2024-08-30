# STAMP: Toward subtask decomposition-based learning and benchmarking for genetic perturbation outcome prediction and beyond
[![DOI image](https://zenodo.org/badge/DOI/10.5281/zenodo.12779567.svg)](https://zenodo.org/records/12779567)
## Introduction 
This repository hosts the official implementation of STAMP, a method that can predict perturbation outcomes using single-cell RNA-sequencing data from perturbational experimental screens using subtask decomposition learning. STAMP can be applied to three challenges in this area, i.e. (1) predict single genetic perturbation outcomes, (2) predict multiple genetic perturbation outcomes and (3) predict genetic perturbation outcomes across cell lines.
<p align="center"><img src="https://github.com/bm2-lab/STAMP/blob/main/img/framework.png" alt="STAMP" width="900px" /></p>  

## Installation
Our experiments were conducted on python=3.9.7 and our CUDA version is 11.4.

We recommend using Anaconda / Miniconda to create a conda environment for using STAMP. You can create a python environment using the following command:
```python
conda  create -n stamp python==3.9.7
```

Then, you can activate the environment using:
```python
conda activate stamp
```
Installing Pytorch with following command:
```python
conda install pytorch==1.10.2 torchvision==0.11.3 torchaudio==0.10.2 -c pytorch
```
Then
```python
pip install .
```
or you can install it from [PyPI](https://pypi.org/project/cell-stamp/):
```
pip install cell-stamp
```

## Example data
We have made available the code necessary to generate example data, serving as a practical illustration for training and testing the STAMP model. Additionally, for guidance on configuring the training process of STAMP, we offer an example config file located at `./Data/example_config.yaml`.
```python
python ./Data/GeneratingExampleData.py
```
The example *.h5ad data file has three distinct layers, namely 'level1', 'level2', and 'level3'. The 'level1' layer is a binary matrix, where '0' represents non-differentially expressed genes (non-DEGs) and '1' indicates differentially expressed genes (DEGs). Similarly, 'level2' is another binary matrix, denoting down-regulated genes with '0' and up-regulated genes with '1'. Lastly, the 'level3' layer is a matrix that quantifies the magnitude of gene expression changes.

## Real demo data
We have uploaded all benchmark datasets to Zenodo, which can be obtained from [here](https://zenodo.org/records/12779567). Please download all these files into the `./Data` directory and refer `tutorial_for_training.py.ipynb` in `./Tutorial` directory. This tutorial uses the one fold of RPE1_essential dataset as an example to perform the model training, testing and check the loss curves during training.
#### Note: Users are encouraged to change the path of each data in 'Config.yaml' based on their own machines.

## Core API interface for model training
Using this API, you can train and test STAMP on your own perturbation datasets using a few lines of code. 
```python
from stamp import STAMP, load_config
import scanpy as sc

# load config file
config = load_config("./Data/example_config.yaml")

# set up and train a STAMP
model = STAMP(config)
model.train()

# load trained model
model.load_pretrained(f"{config['Train']['output_dir']}/trained_models")

# use trained model to predict unseen perturbations
model.prediction(config['dataset']['Testing_dataset'], combo_test = True)

# use trained model to predict unseen perturbations; considering Top 40 DEGs
# Top 40 DEGs consisting of Top 20 up-regulation genes and Top 20 down-regulation genes

# load Top 40 test data
top_40_data = sc.read_h5ad("./Data/example_test_top40.h5ad")

# prediction
model.prediction(top_40_data, combo_test = True)
```
## Core API interface for model fine-tuning
Using this API, you can fine-tune and test STAMP on your own perturbation datasets using a few lines of code.
```python
from stamp import STAMP, load_config
import scanpy as sc

# load config file (we use the example config used for model training to illustrate this)
config = load_config("./Data/example_config.yaml")

# set up STAMP
model = STAMP(config)

# load pre-trained model
model.load_pretrained(f"{config['Train']['output_dir']}/trained_models")

# fine-tuning model
model.finetuning()

# use fine-tuned model to predict unseen perturbations
model.prediction(config['dataset']['Testing_dataset'], combo_test = False)

# use fine-tuned model to predict unseen perturbations; considering Top 40 DEGs
# Top 40 DEGs consisting of Top 20 up-regulation genes and Top 20 down-regulation genes

# load Top 40 test data
top_40_data = sc.read_h5ad("./Data/example_test_top40.h5ad")

# prediction
model.prediction(top_40_data, combo_test = False)
```
## Citation
Yicheng Gao, Zhiting Wei, Qi Liu et al. *Toward subtask decomposition-based learning and benchmarking for genetic perturbation outcome prediction and beyond*, Nature Computational Science, 2024.
## Contacts
bm2-lab@tongji.edu.cn
