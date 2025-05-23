import numpy as np, pandas as pd, scanpy as sc
import os, sys, pickle, joblib, re, torch, subprocess
from itertools import combinations
from collections import Counter
import seaborn as sns
from collections import defaultdict, OrderedDict
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve, roc_auc_score, mean_squared_error
from scipy import sparse
import scipy
import anndata as ad
from tqdm import tqdm
from multiprocessing import Pool


class multidict(dict):
    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value


def convertAnn2Matrix(adata, dirName='./'):
    a  = pd.DataFrame(adata.var)
    a['index'] = a.index
    a = a[['gene_ids', 'index']]
    a.to_csv("{}/genes.tsv".format(dirName),  sep = "\t", index = False, header=False)
    pd.DataFrame(adata.obs.index).to_csv("{}/barcodes.tsv".format(dirName), sep = "\t", index = False, header=False)
    if not sparse.issparse(adata.X): adata.X = sparse.csr_matrix(adata.X) ### 转换为稀疏矩阵
    adata.X = adata.X.astype(np.int32)  ###转换为整数
    scipy.io.mmwrite("{}/matrix.mtx".format(dirName), adata.X.T)




def getsubgroup(x = 'combo_seen0', seed = 1):
    tmp = '../../gears/data/train/splits/train_simulation_{}_0.8_subgroup.pkl'.format(seed)
    with open(tmp, 'rb') as fin:
        subgroup_split = pickle.load(fin)
        test_subgroup = subgroup_split['test_subgroup'][x]
        mylist = []
        if 'combo' in x:
            for i in test_subgroup:
                mylist.append(','.join(sorted(i.split('+'))))
        else:
            for i in test_subgroup:
                mylist.append(i.split('+')[0])
    return mylist


def getsubgroup_single(seed = 1):   ### 单个扰动
    tmp1 = '../../gears/data/train/splits/train_simulation_{}_0.8.pkl'.format(seed)
    tmp2 = '../../../gears/data/train/splits/train_simulation_{}_0.8.pkl'.format(seed)

    if os.path.isfile(tmp1):
        tmp = tmp1
    else:
        tmp = tmp2
    with open(tmp, 'rb') as fin:
        subgroup_split = pickle.load(fin)
        mylist = []
        for i in subgroup_split['test']:
            mylist.append(i.split('+')[0])
    return mylist




def wilcoxonFun(gene):
    sc.tl.rank_genes_groups(adata, 'gene', groups=[gene], reference='CTRL', method= 'wilcoxon')
    result = adata.uns['rank_genes_groups']
    groups = result['names'].dtype.names
    final_result = pd.DataFrame({group + '_' + key: result[key][group] for group in groups for key in ['names', 'pvals_adj', 'logfoldchanges', 'scores']})
    for group in groups:
        tmp1 = group + '_' + 'foldchanges'
        tmp2 = group + '_' + 'logfoldchanges'
        final_result[tmp1] = 2 ** final_result[tmp2]  ### logfoldchange 转换为 foldchange
        final_result.drop(labels=[tmp2], inplace=True, axis=1)
    return final_result

def getDEG(dirName, method='Ttest'):
    os.chdir(dirName)
    global adata
    fileout = '{}_DEG.tsv'.format(method)
    #adata = sc.read_h5ad('filterNor.h5ad')
    adata = sc.read_h5ad('filterNor_subset.h5ad')
    if 'log1p' not in adata.uns:
        adata.uns['log1p'] = {}
    adata.uns['log1p']["base"] = None
    if scipy.sparse.issparse(adata.X):
        adata.X = adata.X.toarray()
    adata.X += .1
    genes = [i for i in set(adata.obs['gene']) if i != 'CTRL']
    if method == 'Ttest':
        sc.tl.rank_genes_groups(adata, 'gene', groups=genes[:], reference='CTRL', method= 't-test')
        result = adata.uns['rank_genes_groups']
        groups = result['names'].dtype.names
        final_result = pd.DataFrame({group + '_' + key: result[key][group] for group in groups for key in ['names', 'pvals_adj', 'logfoldchanges', 'scores']})
        for group in groups:
            tmp1 = group + '_' + 'foldchanges'
            tmp2 = group + '_' + 'logfoldchanges'
            final_result[tmp1] = 2 ** final_result[tmp2]  ### logfoldchange 转换为 foldchange
            final_result.drop(labels=[tmp2], inplace=True, axis=1)
        final_result.sort_index(axis=1, inplace=True)
        final_result.to_csv(fileout, sep='\t', index=False)
    elif method == 'wilcoxon':
        result = myPool(wilcoxonFun, genes, processes=7)
        final_result = pd.concat(result, axis=1)
        final_result.sort_index(axis=1, inplace=True)
        final_result.to_csv(fileout, sep='\t', index=False)




### 根据DEG得到二维矩阵, 列是表达的基因, 行是扰动的基因, 
###   第一级任务。 *****************
def f_getDEG1(dirName, topDeg=0, pvalue = 0.01, method='Ttest'):
    os.chdir(dirName)
    if topDeg == 0:
        filein = '{}_DEG.tsv'.format(method);  fileout = '{}_DEG_binary.tsv'.format(method)
    else:
        filein = '{}_DEG.tsv'.format(method);  fileout = '{}_DEG_binary_topDeg{}.tsv'.format(method, topDeg)
    if os.path.isfile(fileout):
        tmp = pd.read_csv(fileout, sep='\t')
        if tmp.shape[0] >= 10: pass
    dat = pd.read_csv(filein, sep='\t')
    pertGene = list(set([i.split('_')[0] for i in dat.columns]))
    expGene = list(dat.iloc[:, 1])
    binaryMat = pd.DataFrame(columns= expGene, index=pertGene, data=0)
    for pertGene1 in pertGene:
        tmp1 = '{}_names'.format(pertGene1); tmp2 = '{}_pvals_adj'.format(pertGene1); 
        if topDeg == 0:
            expGene1 = [i for i, j in zip(dat[tmp1], dat[tmp2]) if j <= pvalue]
        else:
            expGene1 = list(dat[tmp1][:20]) + list(dat[tmp1][-20:])  ### 上下调各取20个
        binaryMat.loc[pertGene1, expGene1] = 1
    binaryMat.sort_index(axis=0, inplace=True)  ### 排序
    binaryMat.sort_index(axis=1, inplace=True)  ### 排序
    binaryMat.to_csv(fileout, index=True, sep='\t', header=True)


### 分类成上下调  ###   第二级任务。 *****************
def f_getDEG2(dirName, topDeg=0, method='Ttest'):
    os.chdir(dirName)
    if topDeg ==0:   ### 其实没必要，因为预测值不会随着topDeg而改变
        filein = '{}_DEG.tsv'.format(method); fileout = '{}_DEG_UpDown.tsv'.format(method)
    else:
        filein = '{}_DEG.tsv'.format(method);  fileout = '{}_DEG_UpDown_topDeg{}.tsv'.format(method, topDeg)
    if os.path.isfile(fileout):
        tmp = pd.read_csv(fileout, sep='\t')
        if tmp.shape[0] >= 10: pass
    dat = pd.read_csv(filein, sep='\t')
    pertGene = list(set([i.split('_')[0] for i in dat.columns]))
    expGene = list(dat.iloc[:, 1])
    binaryMat = pd.DataFrame(columns= expGene, index=pertGene, data=0)
    for pertGene1 in pertGene:
        tmp1 = '{}_names'.format(pertGene1); tmp2 = '{}_foldchanges'.format(pertGene1)
        expGene1 = [i for i, j in zip(dat[tmp1], dat[tmp2]) if j >= 1]
        binaryMat.loc[pertGene1, expGene1] = 1
    binaryMat.sort_index(axis=0, inplace=True)  ### 排序
    binaryMat.sort_index(axis=1, inplace=True)  ### 排序
    binaryMat.to_csv(fileout, index=True, sep='\t', header=True)


def f_getDEG3(dirName, topDeg = 0, method='Ttest'):
    os.chdir(dirName)
    if topDeg == 0:   ### 其实没必要，因为预测值不会随着topDeg而改变
        filein = '{}_DEG.tsv'.format(method); fileout = '{}_DEG_foldchange.tsv'.format(method)
    else:
        filein = '{}_DEG.tsv'.format(method);  fileout = '{}_DEG_foldchange_topDeg{}.tsv'.format(method, topDeg)
    if os.path.isfile(fileout):
        tmp = pd.read_csv(fileout, sep='\t')
        if tmp.shape[0] >= 10: pass
    dat = pd.read_csv(filein, sep='\t')
    pertGene = list(set([i.split('_')[0] for i in dat.columns]))
    expGene = list(dat.iloc[:, 1])
    expGene = sorted(expGene)
    binaryMat = pd.DataFrame(columns= expGene, index=pertGene, data=0.0)
    for pertGene1 in pertGene:
        tmp1 = '{}_names'.format(pertGene1); tmp2 = '{}_foldchanges'.format(pertGene1)
        tmp3 = dat[[tmp1, tmp2]]
        expGene1 = list(tmp3.sort_values(tmp1)[tmp2])
        binaryMat.loc[pertGene1, :] = expGene1
    binaryMat.sort_index(axis=0, inplace=True)
    binaryMat.sort_index(axis=1, inplace=True)
    binaryMat.to_csv(fileout, index=True, sep='\t', header=True)


def mergeLevelData(dirName, topDeg=0, method='Ttest'):
    os.chdir(dirName)
    if topDeg == 0:
        filein1 = '{}_DEG_binary.tsv'.format(method)
        filein2 = '{}_DEG_UpDown.tsv'.format('Ttest')
        filein3 = '{}_DEG_foldchange.tsv'.format("Ttest")
        fileout = '{}_merge.h5ad'.format(method)
    else:
        filein1 = '{}_DEG_binary_topDeg{}.tsv'.format(method, topDeg)
        filein2 = '{}_DEG_UpDown_topDeg{}.tsv'.format("Ttest", topDeg)
        filein3 = '{}_DEG_foldchange_topDeg{}.tsv'.format("Ttest", topDeg)
        fileout = '{}_merge_topDeg{}.h5ad'.format(method, topDeg)

    
    dat1 = pd.read_csv(filein1, sep='\t', index_col=0)
    dat2 = pd.read_csv(filein2, sep='\t', index_col=0)
    dat2 = dat2.loc[dat1.index, dat1.columns]
    
    dat3 = pd.read_csv(filein3, sep='\t', index_col=0)
    dat3 = dat3.loc[dat1.index, dat1.columns]
    adata = ad.AnnData(X=sparse.csr_matrix(dat1.values), obs=pd.DataFrame(index=dat1.index), var=pd.DataFrame(index=dat1.columns))
    adata.layers['level1'] = sparse.csr_matrix(dat1.values)
    adata.layers['level2'] = sparse.csr_matrix(dat2.values)
    adata.layers['level3'] = sparse.csr_matrix(dat3.values)
    adata.write_h5ad(fileout)



def preData(adata, filterNone=True, minNums = 30, shuffle=True, filterCom=False,  seed = 42, mtpercent = 10,  min_genes = 200, domaxNums=False, doNor=True, min_cells=3):  #### 为了测试聚类，最好不要进行排序
    if domaxNums: maxNums = 50
    adata.var_names.astype(str)
    adata.var_names_make_unique()
    adata = adata[~adata.obs.index.duplicated()]
    if filterCom:
        tmp = adata.obs['gene'].apply(lambda x: True if ',' not in x else False);  adata = adata[tmp]
    if filterNone:
        adata = adata[adata.obs["gene"] != "None"]
    filterNoneNums = adata.shape[0]
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells= min_cells)
    filterCells = adata.shape[0]

    if np.any([True if i.startswith('mt-') else False for i in adata.var_names]):
        adata.var['mt'] = adata.var_names.str.startswith('mt-')
    else:
        adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    if sum(adata.obs['pct_counts_mt'] < 10) / adata.shape[0] <=0.5: mtpercent = 15
    adata = adata[adata.obs.pct_counts_mt < mtpercent, :]
    filterMT = adata.shape[0]
    tmp = adata.obs['gene'].value_counts()  
    tmp_bool = tmp >= minNums
    genes = list(tmp[tmp_bool].index)
    if 'CTRL' not in genes: genes += ['CTRL']
    adata = adata[adata.obs['gene'].isin(genes), :]
    if domaxNums:
        adata1 = adata[adata.obs['gene'] == 'CTRL']
        genes = adata.obs['gene'].unique()
        tmp = [adata[adata.obs['gene'] == i][:maxNums] for i in genes if i !='CTRL']
        adata2 = ad.concat(tmp)
        adata = ad.concat([adata1, adata2])

    filterMinNums = adata.shape[0]
    
    if doNor:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    adata = adata[adata.obs.sort_values(by='gene').index,:]
    if shuffle:
        tmp = list(adata.obs.index)
        np.random.seed(seed); np.random.shuffle(tmp); adata = adata[tmp]
    return filterNoneNums, filterCells, filterMT, filterMinNums, adata



def myPool(func, mylist, processes):
    with Pool(processes) as pool:
        results = list(tqdm(pool.imap(func, mylist), total=len(mylist)))
    return results