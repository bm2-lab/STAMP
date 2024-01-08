import numpy as np
import torch
from torch.utils.data import Dataset
import joblib
import scanpy as sc
import numpy as np

class PerturbDataSet(Dataset):
    def __init__(self, data_dir, gene_embedding_notebook):
        
        # read data
        try:
            data = sc.read_h5ad(data_dir)
        except:
            data = data_dir
        
        # obtain three levels data
        self.FL_data = data.layers['level1']
        self.SL_data = data.layers['level2']
        self.TL_data = data.layers['level3']
        
        # obtain the gene names
        self.gene_name = list(data.var.index)
        
        # obtain perturb genes
        self.perturb_genes = list(data.obs.index)
        
        # obtain gene embedding matrix
        self.gene_embedding_notebook = gene_embedding_notebook
        
    def __getitem__(self, item):
        
        target_gene = self.perturb_genes[item]

        pertub_embeds = 0
        for idx, t in enumerate(target_gene.split(',')):
            target_gene_index = self.gene_name.index(t)
            pertub_embeds += self.gene_embedding_notebook[target_gene_index]
        pertub_embeds /= idx + 1
        
        FL_output = self.FL_data[item].toarray()
        
        SL_output = self.SL_data[item].toarray()
        
        TL_output = self.TL_data[item].toarray()
        
        return (target_gene, pertub_embeds), (FL_output, SL_output, TL_output)
    
    def __len__(self, ):
        return len(self.perturb_genes)

def __main__():
    dataset = PerturbDataSet("Path")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = 64, shuffle = True)
    for idx, batch_x in enumerate(dataloader):
        print(idx, batch_x)