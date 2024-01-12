import torch
from torch import nn
import numpy as np
import math
import matplotlib.pyplot as plt
from torch.nn import functional as F

class Bayes_first_level_layer(nn.Module):
    def __init__(self, in_features, out_features, bias = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma_log = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = bias
        
        if self.bias:
            self.bias_mu = nn.Parameter(torch.Tensor(out_features))
            self.bias_sigma_log = nn.Parameter(torch.Tensor(out_features))
            
        self.activation = nn.Softmax(dim = 1)
        
    def forward(self, X):
        weight = self.weight_mu + torch.randn_like(self.weight_sigma_log) * torch.exp(self.weight_sigma_log)
        if self.bias:
            bias = self.bias + torch.randn_like(self.bias_sigma_log) * torch.exp(self.bias_sigma_log)
        else:
            bias = None
        hidden_states =  F.linear(X, weight, bias)
        output = self.activation(hidden_states)
        return output


class First_level_layer(nn.Module):
    def __init__(self, in_features, out_features, bias = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.mapping1 = nn.ModuleList([nn.Linear(self.in_features, 1024), nn.ReLU(), nn.Dropout(0.9),
                                       nn.Linear(1024, 2048), nn.ReLU(), nn.Dropout(0.9),
                                       nn.Linear(2048, self.out_features)])
            
        self.activation = nn.Sigmoid()

    def forward(self, X):
        hidden_states = X
        for idx, h in enumerate(self.mapping1):
            hidden_states =  h(hidden_states)
        output = self.activation(hidden_states)
        return output
    
class First_level_layer_concate(nn.Module):
    def __init__(self, hid1_features = 128, hid2_features = 64, hid3_features = 32, gene_embedding_notebook = None):
        super().__init__()
        self.hid1_features = hid1_features
        self.hid2_features = hid2_features
        self.hid3_features = hid3_features
        # self.gene_embedding_notebook = nn.Parameter(gene_embedding_notebook)
        self.gene_embedding_notebook = gene_embedding_notebook
        
        self.mapping1 = nn.ModuleList([nn.Linear(self.gene_embedding_notebook.shape[1]*2, self.hid1_features), nn.ReLU(), 
                                       nn.Linear(self.hid1_features, self.hid2_features), nn.ReLU(), 
                                       nn.Linear(self.hid2_features, self.hid3_features), nn.ReLU()] 
                                      )
        self.mapping1_head = nn.Linear(self.hid3_features, 1)
        
        self.activation = nn.Sigmoid()
        
    def forward(self, pertub_genes_embeds):
        
        pertub_genes_embeds = pertub_genes_embeds.unsqueeze(1).expand(pertub_genes_embeds.shape[0],  self.gene_embedding_notebook.shape[0], -1)
        expanded_notebook = self.gene_embedding_notebook.expand(pertub_genes_embeds.shape[0], self.gene_embedding_notebook.shape[0], -1)
        hids = torch.cat([pertub_genes_embeds, expanded_notebook.to(pertub_genes_embeds.device)], dim = -1)
        for idx, h in enumerate(self.mapping1):
            hids = h(hids)
        hids_head = self.mapping1_head(hids)
        output = self.activation(hids_head)
        
        return output

class Second_level_layer(nn.Module):
    def __init__(self, hid1_features = 128, hid2_features = 64, hid3_features = 32, gene_embedding_notebook = None):
        super().__init__()
        self.hid1_features = hid1_features
        self.hid2_features = hid2_features
        self.hid3_features = hid3_features
        # self.gene_embedding_notebook = nn.Parameter(gene_embedding_notebook)
        self.gene_embedding_notebook = gene_embedding_notebook
        
        self.mapping2 = nn.ModuleList([nn.Linear(self.gene_embedding_notebook.shape[1] * 2, self.hid1_features), nn.ReLU(), 
                                       nn.Linear(self.hid1_features, self.hid2_features), nn.ReLU(), 
                                       nn.Linear(self.hid2_features, self.hid3_features), nn.ReLU()] 
                                      )
        self.mapping2_head = nn.Linear(self.hid3_features, 1)
        
        self.activation = nn.Sigmoid()
        
    def forward(self, X, pertub_genes_embeds):
        
        with torch.no_grad():
            mask = X==0
        pertub_genes_embeds = pertub_genes_embeds.unsqueeze(1).expand(pertub_genes_embeds.shape[0],  self.gene_embedding_notebook.shape[0], -1)
        expanded_notebook = self.gene_embedding_notebook.expand(pertub_genes_embeds.shape[0], self.gene_embedding_notebook.shape[0], -1)
        hids = torch.cat([pertub_genes_embeds, expanded_notebook.to(X.device)], dim = -1)
        for idx, h in enumerate(self.mapping2):
            hids = h(hids)
        hids_head = self.mapping2_head(hids)
        output_second_level = self.activation(hids_head)
        
        return output_second_level, ~mask, hids

class Third_level_layer(nn.Module):
    def __init__(self, in_features = 32, hid1_features = 16, hid2_features = 8):
        super().__init__()
        self.in_features = in_features
        self.hid1_features = hid1_features
        self.hid2_features = hid2_features
        self.mapping3 = nn.ModuleList([nn.Linear(in_features, hid1_features), 
                                       nn.Linear(hid1_features, hid2_features)])
        self.mapping3_head = nn.Linear(hid2_features, 1)
        self.activation = nn.LeakyReLU()
    
    def forward(self, X, mask = None):
        hids = X
        for idx, h in enumerate(self.mapping3):
            hids = h(hids)
        hids_head = self.mapping3_head(hids)
        # output_third_level = self.activation(hids_head)
        output_third_level = torch.exp(hids_head)
        return output_third_level, mask

class TaskCombineLayer_multi_task(nn.Module):
    def __init__(self, in_features, out_features, 
                 hid1_features_2 = 128, hid2_features_2 = 64, hid3_features_2 = 32,
                 in_feature_3 = 32, hid1_features_3 = 16, hid2_features_3 = 8, 
                 gene_embedding_notebook = None ,bias = True):
        super().__init__()
        
        # you can set gene embeddings as a learnable parameters
        # self.gene_embedding_notebook = nn.Parameter(gene_embedding_notebook)
        self.gene_embedding_notebook = gene_embedding_notebook
        self.first_level_layer = First_level_layer(in_features, out_features)
        self.second_level_layer = Second_level_layer(hid1_features_2, hid2_features_2, hid3_features_2, gene_embedding_notebook = self.gene_embedding_notebook)
        self.third_level_layer = Third_level_layer(in_feature_3, hid1_features_3, hid2_features_3)
        
    def forward(self, X, 
                level_1_output, 
                level_2_output, 
                level_3_output
                ) :
        output_1 = self.first_level_layer(X)
        # binary loss
        with torch.no_grad():
            all_num = level_1_output.shape[0] * level_1_output.shape[1]
            DE_num = torch.sum(level_1_output.sum())
            if DE_num == 0:
                DE_num = 1
            loss_weights = (all_num - DE_num) / DE_num
        loss_binary = nn.BCELoss(weight = level_1_output * loss_weights / 4 + 1)
        loss_1 = loss_binary(output_1.squeeze(-1), level_1_output)

        output_2, mask, hids = self.second_level_layer(level_1_output, X)
        with torch.no_grad():
            up_num = torch.sum(mask * level_2_output)
            all_num = torch.sum(mask)
            if up_num == 0:
                up_num = 1
            weights = mask * level_2_output * all_num / up_num
            weights[weights > 0] -= 1
            new_weights = weights + mask
            if all_num <= up_num:
                all_num = up_num + 1
            new_weights[new_weights==1] = all_num /(all_num - up_num)

        loss_binary = nn.BCELoss(weight = new_weights, reduction = 'sum')
        loss_2 = loss_binary(output_2.squeeze(-1), level_2_output) / torch.sum(new_weights)

        output_3, mask = self.third_level_layer(hids, mask)
        loss_3 = torch.sum((mask * (output_3.squeeze(-1)-level_3_output) ** 2)) / torch.sum(mask)
        
        return loss_1, loss_2, loss_3, (output_1, output_2, output_3), mask

class TaskCombineLayer_multi_task_concate(nn.Module):
    def __init__(self, hid1_features_1 = 128, hid2_features_1 = 64, hid3_features_1 = 32,
                 hid1_features_2 = 128, hid2_features_2 = 64, hid3_features_2 = 32,
                 in_feature_3 = 32, hid1_features_3 = 16, hid2_features_3 = 8, 
                 gene_embedding_notebook = None ,bias = True):
        super().__init__()
        
        # self.gene_embedding_notebook = nn.Parameter(gene_embedding_notebook)
        self.gene_embedding_notebook = gene_embedding_notebook
        self.first_level_layer = First_level_layer_concate(hid1_features_1, hid2_features_1, hid3_features_1, gene_embedding_notebook = self.gene_embedding_notebook)
        self.second_level_layer = Second_level_layer(hid1_features_2, hid2_features_2, hid3_features_2, gene_embedding_notebook = self.gene_embedding_notebook)
        self.third_level_layer = Third_level_layer(in_feature_3, hid1_features_3, hid2_features_3)
        
    def forward(self, X, 
                level_1_output, 
                level_2_output, 
                level_3_output
                ) :
        output_1 = self.first_level_layer(X)
        # binary loss
        with torch.no_grad():
            all_num = level_1_output.shape[0] * level_1_output.shape[1]
            DE_num = torch.sum(level_1_output.sum())
            if DE_num == 0:
                DE_num = 1
            loss_weights = (all_num - DE_num) / DE_num
        loss_binary = nn.BCELoss(weight = level_1_output * loss_weights / 4 + 1)
        loss_1 = loss_binary(output_1.squeeze(-1), level_1_output)
        
        output_2, mask, hids = self.second_level_layer(level_1_output, X)
        with torch.no_grad():
            up_num = torch.sum(mask * level_2_output)
            all_num = torch.sum(mask)
            if up_num == 0:
                up_num = 1
            weights = mask * level_2_output * all_num / up_num
            weights[weights > 0] -= 1
            new_weights = weights + mask
            if all_num <= up_num:
                all_num = up_num + 1
            new_weights[new_weights==1] = all_num /(all_num - up_num)

        loss_binary = nn.BCELoss(weight = new_weights, reduction = 'sum')
        loss_2 = loss_binary(output_2.squeeze(-1), level_2_output) / torch.sum(new_weights)

        output_3, mask = self.third_level_layer(hids, mask)
        loss_3 = torch.sum((mask * (output_3.squeeze(-1)-level_3_output) ** 2)) / torch.sum(mask)
        
        return loss_1, loss_2, loss_3, (output_1, output_2, output_3), mask