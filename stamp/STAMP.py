import scanpy as sc
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
import torch
from torch import nn
import numpy as np
import math
import matplotlib.pyplot as plt
import time
import random
import joblib
import pandas as pd
from .DataSet import PerturbDataSet
from torch.nn import functional as F
from .Modules import TaskCombineLayer_multi_task
import logging
import yaml
import pickle
from copy import deepcopy
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from scipy.stats import pearsonr
import anndata as ad

class STAMP:
    """
    STAMP base model class
    """    
    
    def __init__(self, 
                 config = None):
        assert config is not None
        torch.manual_seed(config['Train']['Trainer_parameter']['random_seed'])
        torch.cuda.manual_seed_all(config['Train']['Trainer_parameter']['random_seed'])
        np.random.seed(config['Train']['Trainer_parameter']['random_seed'])
        random.seed(config['Train']['Trainer_parameter']['random_seed'])
        torch.cuda.manual_seed(config['Train']['Trainer_parameter']['random_seed'])
        self.config = config
        
        self.device = config['Train']['Model_Parameter']['device']
        with open(config['dataset']['Gene_embedding'], 'rb') as fin:
            # Gene_embeddings = pickle.load(fin)
            Gene_embeddings = joblib.load(fin)
        self.Gene_embeddings = torch.tensor(Gene_embeddings)
        self.model = TaskCombineLayer_multi_task(in_features = self.config['Train']['Model_Parameter']['First_level']['in_features'], 
                                    out_features = self.config['Train']['Model_Parameter']['First_level']['out_features'], 
                                    hid1_features_2 = self.config['Train']['Model_Parameter']['Second_level']['hid1_features_2'], 
                                    hid2_features_2 = self.config['Train']['Model_Parameter']['Second_level']['hid2_features_2'], 
                                    hid3_features_2 = self.config['Train']['Model_Parameter']['Second_level']['hid3_features_2'], 
                                    in_feature_3 = self.config['Train']['Model_Parameter']['Third_level']['in_feature_3'], 
                                    hid1_features_3 = self.config['Train']['Model_Parameter']['Third_level']['hid1_features_3'], 
                                    hid2_features_3 = self.config['Train']['Model_Parameter']['Third_level']['hid2_features_3'], 
                                    gene_embedding_notebook = self.Gene_embeddings, bias = True
                                    ).to(self.device)

        self.best_model_firstlevel = deepcopy(self.model.first_level_layer)
        self.best_model_secondthirdlevel = deepcopy(self.model)

    def get_logger(self, filename, verbosity=1, name=None):
        level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
        formatter = logging.Formatter(
            "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
        )
        logger = logging.getLogger(name)
        logger.setLevel(level_dict[verbosity])

        fh = logging.FileHandler(filename, "w")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)

        return logger

    def train(self, ):
        
        dataset = PerturbDataSet(self.config['dataset']['Training_dataset'], self.Gene_embeddings)
        dataset_val = PerturbDataSet(self.config['dataset']['Validation_dataseta'], self.Gene_embeddings)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = self.config['Train']['Sampling']['batch_size'], shuffle = self.config['Train']['Sampling']['sample_shuffle'])
        dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size = self.config['Inference']['Sampling']['batch_size'], shuffle = self.config['Inference']['Sampling']['sample_shuffle'])
        
        min_val_first_level = np.inf
        min_val_third_level = np.inf
        
        best_model = deepcopy(self.model)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr = self.config['Train']['Trainer_parameter']['learning_rate'])
        epochs = self.config['Train']['Trainer_parameter']['epoch']

        logger = self.get_logger(self.config['Train']['output_dir']+'/training.log')
        
        # training and testing
        for epoch in range(1, epochs + 1):
            loss1_train = 0
            loss2_train = 0
            loss3_train = 0
            s = time.time()
            for idx, batch_x in enumerate(dataloader):
                if torch.sum(batch_x[1][0].squeeze(1).float()) == 0:
                    continue
                losses = self.model(batch_x[0][1].to(self.device), 
                            batch_x[1][0].squeeze(1).float().to(self.device), 
                            batch_x[1][1].squeeze(1).float().to(self.device), 
                            batch_x[1][2].squeeze(1).float().to(self.device))
                optimizer.zero_grad()
                loss = 0.1 * losses[0] + 0.5 * losses[1] + 0.4 * losses[2]
                loss.backward()
                optimizer.step()
                loss1_train += losses[0].item()
                loss2_train += losses[1].item()
                loss3_train += losses[2].item()
            e = time.time()
            loss1_train /= (idx + 1)
            loss2_train /= (idx + 1)
            loss3_train /= (idx + 1)
            
            loss1_test = 0
            loss2_test = 0
            loss3_test = 0
            with torch.no_grad():
                for idx_test, batch_x_test in enumerate(dataloader_val):
                    losses_test = self.model.eval()(batch_x_test[0][1].to(self.device), 
                                                batch_x_test[1][0].squeeze(1).float().to(self.device), 
                                                batch_x_test[1][1].squeeze(1).float().to(self.device), 
                                                batch_x_test[1][2].squeeze(1).float().to(self.device))
                    loss1_test += losses_test[0].item()
                    loss2_test += losses_test[1].item()
                    loss3_test += losses_test[2].item()
            loss1_test /= (idx_test + 1)
            loss2_test /= (idx_test + 1)
            loss3_test /= (idx_test + 1)
            logger.info('Epoch:[{}/{}]\tsteps:{}\tloss1_train:{:.5f}\tloss2_train:{:.5f}\tloss3_train:{:.5f}\tloss1_test:{:.5f}\tloss2_test:{:.5f}\tloss3_test:{:.5f}\ttime:{:.3f}'.format(epoch, epochs, idx + 1, 
                                                                                                                                                                                           loss1_train, loss2_train, loss3_train, 
                                                                                                                                                                                           loss1_test, loss2_test, loss3_test, e-s))
            
            if loss1_test < min_val_first_level:
                min_val_first_level = loss1_test
                best_model_firstlevel = deepcopy(self.model.first_level_layer)
            if loss3_test < min_val_third_level:
                min_val_third_level = loss3_test
                best_model_secondthirdlevel = deepcopy(self.model)
                                                                                                                                                                               
        logger.info('finish training!')
        self.best_model_firstlevel = best_model_firstlevel
        self.best_model_secondthirdlevel = best_model_secondthirdlevel
        if not os.path.exists(self.config['Train']['output_dir'] + '/trained_models'):
            os.mkdir(self.config['Train']['output_dir'] + '/trained_models')
        torch.save({'model_state_dict':self.best_model_firstlevel.state_dict()},f"{self.config['Train']['output_dir']}/trained_models/firstlevel_model.pth")
        torch.save({'model_state_dict':self.best_model_secondthirdlevel.state_dict()},f"{self.config['Train']['output_dir']}/trained_models/secondthirdlevel_model.pth")
    
    def load_pretrained(self, model_path = None):
        assert model_path is not None
        checkpoint_first_level = torch.load(model_path + '/firstlevel_model.pth')
        checkpoint_secondthird_level = torch.load(model_path + '/secondthirdlevel_model.pth')
        self.model.load_state_dict(checkpoint_secondthird_level['model_state_dict'])
        self.model.first_level_layer.load_state_dict(checkpoint_first_level['model_state_dict'])
        self.best_model_firstlevel = self.model.first_level_layer
        self.best_model_secondthirdlevel = self.model
    
    def frozen_layers(self, ):
        for param in self.model.first_level_layer.mapping1[0].parameters():
            param.requires_grad = False
        for param in self.model.first_level_layer.mapping1[3].parameters():
            param.requires_grad = False
        for param in self.model.second_level_layer.mapping2[0].parameters():
            param.requires_grad = False    
        for param in self.model.second_level_layer.mapping2[2].parameters():
            param.requires_grad = False    
        # for param in self.model.second_level_layer.mapping2[4].parameters():
        #     param.requires_grad = False  
        # for param in self.model.third_level_layer.mapping3[0].parameters():
        #     param.requires_grad = False          
        # for param in self.model.third_level_layer.mapping3[1].parameters():
        #     param.requires_grad = False    

    def finetuning(self, ):
        
        dataset = PerturbDataSet(self.config['dataset']['Training_dataset'], self.Gene_embeddings)
        dataset_val = PerturbDataSet(self.config['dataset']['Validation_dataseta'], self.Gene_embeddings)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = self.config['Train']['Sampling']['batch_size'], shuffle = self.config['Train']['Sampling']['sample_shuffle'])
        dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size = self.config['Inference']['Sampling']['batch_size'], shuffle = self.config['Inference']['Sampling']['sample_shuffle'])
        
        min_val_first_level = np.inf
        min_val_third_level = np.inf
        
        best_model = deepcopy(self.model)
        optimizer = torch.optim.Adam(self.model.parameters(), lr = self.config['Train']['Trainer_parameter']['learning_rate'])
        epochs = self.config['Train']['Trainer_parameter']['epoch']

        logger = self.get_logger(self.config['Train']['output_dir']+'/training.log')
        
        # training and testing
        for epoch in range(1, epochs + 1):
            loss1_train = 0
            loss2_train = 0
            loss3_train = 0
            s = time.time()
            for idx, batch_x in enumerate(dataloader):
                if torch.sum(batch_x[1][0].squeeze(1).float()) == 0:
                    continue
                losses = self.model(batch_x[0][1].to(self.device), 
                            batch_x[1][0].squeeze(1).float().to(self.device), 
                            batch_x[1][1].squeeze(1).float().to(self.device), 
                            batch_x[1][2].squeeze(1).float().to(self.device))
                optimizer.zero_grad()
                loss = 0.1 * losses[0] + 0.5 * losses[1] + 0.4 * losses[2]
                loss.backward()
                optimizer.step()
                loss1_train += losses[0].item()
                loss2_train += losses[1].item()
                loss3_train += losses[2].item()
            e = time.time()
            loss1_train /= (idx + 1)
            loss2_train /= (idx + 1)
            loss3_train /= (idx + 1)
            
            loss1_test = 0
            loss2_test = 0
            loss3_test = 0
            with torch.no_grad():
                for idx_test, batch_x_test in enumerate(dataloader_val):
                    losses_test = self.model.eval()(batch_x_test[0][1].to(self.device), 
                                                batch_x_test[1][0].squeeze(1).float().to(self.device), 
                                                batch_x_test[1][1].squeeze(1).float().to(self.device), 
                                                batch_x_test[1][2].squeeze(1).float().to(self.device))
                    loss1_test += losses_test[0].item()
                    loss2_test += losses_test[1].item()
                    loss3_test += losses_test[2].item()
            loss1_test /= (idx_test + 1)
            loss2_test /= (idx_test + 1)
            loss3_test /= (idx_test + 1)
            logger.info('Epoch:[{}/{}]\tsteps:{}\tloss1_train:{:.5f}\tloss2_train:{:.5f}\tloss3_train:{:.5f}\tloss1_test:{:.5f}\tloss2_test:{:.5f}\tloss3_test:{:.5f}\ttime:{:.3f}'.format(epoch, epochs, idx + 1, 
                                                                                                                                                                                           loss1_train, loss2_train, loss3_train, 
                                                                                                                                                                                           loss1_test, loss2_test, loss3_test, e-s))
            
            if loss1_test < min_val_first_level:
                min_val_first_level = loss1_test
                best_model_firstlevel = deepcopy(self.model.first_level_layer)
            if loss3_test < min_val_third_level:
                min_val_third_level = loss3_test
                best_model_secondthirdlevel = deepcopy(self.model)
                                                                                                                                                                               
        logger.info('finish training!')
        self.best_model_firstlevel = best_model_firstlevel
        self.best_model_secondthirdlevel = best_model_secondthirdlevel
        if not os.path.exists(self.config['Train']['output_dir'] + '/trained_models'):
            os.mkdir(self.config['Train']['output_dir'] + '/trained_models')
        torch.save({'model_state_dict':self.best_model_firstlevel.state_dict()},f"{self.config['Train']['output_dir']}/trained_models/firstlevel_model.pth")
        torch.save({'model_state_dict':self.best_model_secondthirdlevel.state_dict()},f"{self.config['Train']['output_dir']}/trained_models/secondthirdlevel_model.pth")
    
    def prediction(self, test_file_path, combo_test = False):
        dataset_test = PerturbDataSet(test_file_path, self.Gene_embeddings)
        dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size = 100, shuffle = False)
        output1 = []
        output2 = []
        output3 = []
        labels1 = []
        labels2 = []
        labels3 = []
        with torch.no_grad():
            for idx_test, batch_x_test in enumerate(dataloader_test):
                output_1 = self.best_model_firstlevel.eval()(batch_x_test[0][1].to(self.device))
                output1.append(output_1.cpu())
                labels1.append(batch_x_test[1][0].squeeze(1).float())
                output_2, mask, hids = self.best_model_secondthirdlevel.second_level_layer.eval()(batch_x_test[1][0].squeeze(1).float().to(self.device), batch_x_test[0][1].to(self.device))
                output2.append(output_2.squeeze(-1).cpu())
                labels2.append(batch_x_test[1][1].squeeze(1).float())
                output_3, mask = self.best_model_secondthirdlevel.third_level_layer.eval()(hids, mask)
                output3.append(output_3.squeeze(-1).cpu())
                labels3.append(batch_x_test[1][2].squeeze(1).float())
            output1 = torch.cat(output1, dim = 0)
            output2 = torch.cat(output2, dim = 0)
            output3 = torch.cat(output3, dim = 0)
            labels1 = torch.cat(labels1, dim = 0)
            labels2 = torch.cat(labels2, dim = 0)
            labels3 = torch.cat(labels3, dim = 0)

        try:
            ROCAUC_1 = roc_auc_score(labels1.reshape(-1), output1.reshape(-1))
            lr_precision1, lr_recall1, _ = precision_recall_curve(labels1.reshape(-1), output1.reshape(-1))
            PRAUC_1 = auc(lr_recall1, lr_precision1)
        except:
            ROCAUC_1 = "Nan"
            PRAUC_1 = "Nan"

        try:
            ROCAUC_2 = roc_auc_score(labels2[labels1 == 1], output2[labels1 == 1])
            lr_precision2, lr_recall2, _ = precision_recall_curve(labels2[labels1 == 1], output2[labels1 == 1])
            PRAUC_2 = auc(lr_recall2, lr_precision2)
        except:
            ROCAUC_2 = 'Nan'
            PRAUC_2 = 'Nan'
            
        correlation_3 = pearsonr(output3[labels1 == 1], labels3[labels1 == 1])[0]
        mse_3 = torch.mean((output3[labels1 == 1] - labels3[labels1 == 1])**2)
        print(f"ROC_AUC_1: {ROCAUC_1}, PR_AUC_1: {PRAUC_1}, ROC_AUC_2: {ROCAUC_2}, PR_AUC_2: {PRAUC_2}, COR_3: {correlation_3}, mse_3: {mse_3.item()}")

        d = sc.read_h5ad(self.config['dataset']['Testing_dataset'])

        output_test = ad.AnnData(d.X.toarray())
        output_test.layers['level1'] = np.array(output1)
        output_test.layers['level2'] = np.array(output2)
        output_test.layers['level3'] = np.array(output3)
        output_test.obs = d.obs
        output_test.write(f"{self.config['Train']['output_dir']}/output_test.h5ad")
        
        results = {"Overall":{"ROC_AUC_1": ROCAUC_1, "PR_AUC_1": PRAUC_1, "ROC_AUC_2": ROCAUC_2, "PR_AUC_2": PRAUC_2, "COR_3": correlation_3, "mse_3": mse_3.item()}}
        
        if combo_test:
            
            train_genes = list(sc.read_h5ad(self.config['dataset']['Training_dataset']).obs.index)
            train_genes_flattens = []
            for i in train_genes:
                tmp = i.split(",")
                for j in tmp:
                    train_genes_flattens.append(j)
            
            single_ind = np.array([idx for idx, i in enumerate(list(output_test.obs.index)) if len(i.split(','))==1])
            two_seen_ind = np.array([idx for idx, i in enumerate(list(output_test.obs.index)) if len(i.split(','))>1 and len(set(i.split(',')) & set(train_genes_flattens)) == 2])
            one_seen_ind = np.array([idx for idx, i in enumerate(list(output_test.obs.index)) if len(i.split(','))>1 and len(set(i.split(',')) & set(train_genes_flattens)) == 1])
            zero_seen_ind = np.array([idx for idx, i in enumerate(list(output_test.obs.index)) if len(i.split(','))>1 and len(set(i.split(',')) & set(train_genes_flattens)) == 0])
            assert len(single_ind) + len(two_seen_ind) + len(one_seen_ind) + len(zero_seen_ind) == len(output_test)
            
            ## single_ind
            try:
                ROCAUC_1 = roc_auc_score(labels1[single_ind].reshape(-1), output1[single_ind].reshape(-1))
                lr_precision1, lr_recall1, _ = precision_recall_curve(labels1[single_ind].reshape(-1), output1[single_ind].reshape(-1))
                PRAUC_1 = auc(lr_recall1, lr_precision1)

                ROCAUC_2 = roc_auc_score(labels2[single_ind][labels1[single_ind] == 1], output2[single_ind][labels1[single_ind] == 1])
                lr_precision2, lr_recall2, _ = precision_recall_curve(labels2[single_ind][labels1[single_ind] == 1], output2[single_ind][labels1[single_ind] == 1])
                PRAUC_2 = auc(lr_recall2, lr_precision2)
                
                correlation_3 = pearsonr(output3[single_ind][labels1[single_ind] == 1], labels3[single_ind][labels1[single_ind] == 1])[0]
                mse_3 = torch.mean((output3[single_ind][labels1[single_ind] == 1] - labels3[single_ind][labels1[single_ind] == 1])**2)
                print(f"Single_Perturbation: ROC_AUC_1: {ROCAUC_1}, PR_AUC_1: {PRAUC_1}, ROC_AUC_2: {ROCAUC_2}, PR_AUC_2: {PRAUC_2}, COR_3: {correlation_3}, mse_3: {mse_3.item()}")
                results['Single_Perturbation'] = {"ROC_AUC_1": ROCAUC_1, "PR_AUC_1": PRAUC_1, "ROC_AUC_2": ROCAUC_2, "PR_AUC_2": PRAUC_2, "COR_3": correlation_3, "mse_3": mse_3.item()}
            except:
                ROCAUC_1 = 'Nan'; PRAUC_1 = 'Nan'; ROC_AUC_2 = 'Nan'; PR_AUC_2 = 'Nan'; correlation_3 = 'Nan'; mse_3 = 'Nan'
                print("There is no single perturbations in the test data")
            
            ## two_seen
            try:
                ROCAUC_1 = roc_auc_score(labels1[two_seen_ind].reshape(-1), output1[two_seen_ind].reshape(-1))
                lr_precision1, lr_recall1, _ = precision_recall_curve(labels1[two_seen_ind].reshape(-1), output1[two_seen_ind].reshape(-1))
                PRAUC_1 = auc(lr_recall1, lr_precision1)

                ROCAUC_2 = roc_auc_score(labels2[two_seen_ind][labels1[two_seen_ind] == 1], output2[two_seen_ind][labels1[two_seen_ind] == 1])
                lr_precision2, lr_recall2, _ = precision_recall_curve(labels2[two_seen_ind][labels1[two_seen_ind] == 1], output2[two_seen_ind][labels1[two_seen_ind] == 1])
                PRAUC_2 = auc(lr_recall2, lr_precision2)
                
                correlation_3 = pearsonr(output3[two_seen_ind][labels1[two_seen_ind] == 1], labels3[two_seen_ind][labels1[two_seen_ind] == 1])[0]
                mse_3 = torch.mean((output3[two_seen_ind][labels1[two_seen_ind] == 1] - labels3[two_seen_ind][labels1[two_seen_ind] == 1])**2)
                print(f"Two_seen_Perturbation: ROC_AUC_1: {ROCAUC_1}, PR_AUC_1: {PRAUC_1}, ROC_AUC_2: {ROCAUC_2}, PR_AUC_2: {PRAUC_2}, COR_3: {correlation_3}, mse_3: {mse_3.item()}")
                results['Two_seen_Perturbation'] = {"ROC_AUC_1": ROCAUC_1, "PR_AUC_1": PRAUC_1, "ROC_AUC_2": ROCAUC_2, "PR_AUC_2": PRAUC_2, "COR_3": correlation_3, "mse_3": mse_3.item()}
            except:
                ROCAUC_1 = 'Nan'; PRAUC_1 = 'Nan'; ROC_AUC_2 = 'Nan'; PR_AUC_2 = 'Nan'; correlation_3 = 'Nan'; mse_3 = 'Nan'
                print("There is no 2/2 seen perturbations in the test data")            
            
            ## one_seen
            try:
                ROCAUC_1 = roc_auc_score(labels1[one_seen_ind].reshape(-1), output1[one_seen_ind].reshape(-1))
                lr_precision1, lr_recall1, _ = precision_recall_curve(labels1[one_seen_ind].reshape(-1), output1[one_seen_ind].reshape(-1))
                PRAUC_1 = auc(lr_recall1, lr_precision1)

                ROCAUC_2 = roc_auc_score(labels2[one_seen_ind][labels1[one_seen_ind] == 1], output2[one_seen_ind][labels1[one_seen_ind] == 1])
                lr_precision2, lr_recall2, _ = precision_recall_curve(labels2[one_seen_ind][labels1[one_seen_ind] == 1], output2[one_seen_ind][labels1[one_seen_ind] == 1])
                PRAUC_2 = auc(lr_recall2, lr_precision2)
                
                correlation_3 = pearsonr(output3[one_seen_ind][labels1[one_seen_ind] == 1], labels3[one_seen_ind][labels1[one_seen_ind] == 1])[0]
                mse_3 = torch.mean((output3[one_seen_ind][labels1[one_seen_ind] == 1] - labels3[one_seen_ind][labels1[one_seen_ind] == 1])**2)
                print(f"One_seen_Perturbation: ROC_AUC_1: {ROCAUC_1}, PR_AUC_1: {PRAUC_1}, ROC_AUC_2: {ROCAUC_2}, PR_AUC_2: {PRAUC_2}, COR_3: {correlation_3}, mse_3: {mse_3.item()}")
                results['One_seen_Perturbation'] = {"ROC_AUC_1": ROCAUC_1, "PR_AUC_1": PRAUC_1, "ROC_AUC_2": ROCAUC_2, "PR_AUC_2": PRAUC_2, "COR_3": correlation_3, "mse_3": mse_3.item()}            

            except:
                ROCAUC_1 = 'Nan'; PRAUC_1 = 'Nan'; ROC_AUC_2 = 'Nan'; PR_AUC_2 = 'Nan'; correlation_3 = 'Nan'; mse_3 = 'Nan'
                print("There is no 1/2 seen perturbations in the test data")            
            
            ## zero_seen
            try:
                ROCAUC_1 = roc_auc_score(labels1[zero_seen_ind].reshape(-1), output1[zero_seen_ind].reshape(-1))
                lr_precision1, lr_recall1, _ = precision_recall_curve(labels1[zero_seen_ind].reshape(-1), output1[zero_seen_ind].reshape(-1))
                PRAUC_1 = auc(lr_recall1, lr_precision1)

                ROCAUC_2 = roc_auc_score(labels2[zero_seen_ind][labels1[zero_seen_ind] == 1], output2[zero_seen_ind][labels1[zero_seen_ind] == 1])
                lr_precision2, lr_recall2, _ = precision_recall_curve(labels2[zero_seen_ind][labels1[zero_seen_ind] == 1], output2[zero_seen_ind][labels1[zero_seen_ind] == 1])
                PRAUC_2 = auc(lr_recall2, lr_precision2)
                
                correlation_3 = pearsonr(output3[zero_seen_ind][labels1[zero_seen_ind] == 1], labels3[zero_seen_ind][labels1[zero_seen_ind] == 1])[0]
                mse_3 = torch.mean((output3[zero_seen_ind][labels1[zero_seen_ind] == 1] - labels3[zero_seen_ind][labels1[zero_seen_ind] == 1])**2)
                print(f"Zero_seen_Perturbation: ROC_AUC_1: {ROCAUC_1}, PR_AUC_1: {PRAUC_1}, ROC_AUC_2: {ROCAUC_2}, PR_AUC_2: {PRAUC_2}, COR_3: {correlation_3}, mse_3: {mse_3.item()}")
                results['Zero_seen_Perturbation'] = {"ROC_AUC_1": ROCAUC_1, "PR_AUC_1": PRAUC_1, "ROC_AUC_2": ROCAUC_2, "PR_AUC_2": PRAUC_2, "COR_3": correlation_3, "mse_3": mse_3.item()}            
                
            except:
                ROCAUC_1 = 'Nan'; PRAUC_1 = 'Nan'; ROC_AUC_2 = 'Nan'; PR_AUC_2 = 'Nan'; correlation_3 = 'Nan'; mse_3 = 'Nan'
                print("There is no 0/2 seen perturbations in the test data")            

        return results
    
    def prediction_cross_cell_line(self, test_file_path, cross_cell_line = True, new_cell_line_embedding = None, combo_test = False):
        if cross_cell_line:
            assert new_cell_line_embedding is not None
            with open(new_cell_line_embedding, 'rb') as fin:
                Gene_embeddings = pickle.load(fin)
            Gene_embeddings = torch.tensor(Gene_embeddings)
            dataset_test = PerturbDataSet(test_file_path, Gene_embeddings)
        else:
            dataset_test = PerturbDataSet(test_file_path, self.Gene_embeddings)
        dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size = 100, shuffle = False)
        output1 = []
        output2 = []
        output3 = []
        labels1 = []
        labels2 = []
        labels3 = []
        with torch.no_grad():
            for idx_test, batch_x_test in enumerate(dataloader_test):
                output_1 = self.best_model_firstlevel.eval()(batch_x_test[0][1].to(self.device))
                output1.append(output_1.cpu())
                labels1.append(batch_x_test[1][0].squeeze(1).float())
                output_2, mask, hids = self.best_model_secondthirdlevel.second_level_layer.eval()(batch_x_test[1][0].squeeze(1).float().to(self.device), batch_x_test[0][1].to(self.device))
                output2.append(output_2.squeeze(-1).cpu())
                labels2.append(batch_x_test[1][1].squeeze(1).float())
                output_3, mask = self.best_model_secondthirdlevel.third_level_layer.eval()(hids, mask)
                output3.append(output_3.squeeze(-1).cpu())
                labels3.append(batch_x_test[1][2].squeeze(1).float())
            output1 = torch.cat(output1, dim = 0)
            output2 = torch.cat(output2, dim = 0)
            output3 = torch.cat(output3, dim = 0)
            labels1 = torch.cat(labels1, dim = 0)
            labels2 = torch.cat(labels2, dim = 0)
            labels3 = torch.cat(labels3, dim = 0)

        ROCAUC_1 = roc_auc_score(labels1.reshape(-1), output1.reshape(-1))
        lr_precision1, lr_recall1, _ = precision_recall_curve(labels1.reshape(-1), output1.reshape(-1))
        PRAUC_1 = auc(lr_recall1, lr_precision1)

        ROCAUC_2 = roc_auc_score(labels2[labels1 == 1], output2[labels1 == 1])
        lr_precision2, lr_recall2, _ = precision_recall_curve(labels2[labels1 == 1], output2[labels1 == 1])
        PRAUC_2 = auc(lr_recall2, lr_precision2)
        
        correlation_3 = pearsonr(output3[labels1 == 1], labels3[labels1 == 1])[0]
        mse_3 = torch.mean((output3[labels1 == 1] - labels3[labels1 == 1])**2)
        print(f"ROC_AUC_1: {ROCAUC_1}, PR_AUC_1: {PRAUC_1}, ROC_AUC_2: {ROCAUC_2}, PR_AUC_2: {PRAUC_2}, COR_3: {correlation_3}, mse_3: {mse_3.item()}")

        d = sc.read_h5ad(self.config['dataset']['Testing_dataset'])

        output_test = ad.AnnData(d.X.toarray())
        output_test.layers['level1'] = np.array(output1)
        output_test.layers['level2'] = np.array(output2)
        output_test.layers['level3'] = np.array(output3)
        output_test.obs = d.obs
        output_test.write(f"{self.config['Train']['output_dir']}/output_test.h5ad")
        