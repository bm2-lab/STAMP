# STAMP
## Introduction 
## Core API interface
```python
from STAMP import STAMP
import scanpy as sc
import yaml

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
