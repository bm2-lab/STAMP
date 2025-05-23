import sys
sys.path.append('/home//project/GW_PerturbSeq')
from myUtil import *
import scanpy as sc
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

### 数据预处理 以及得到True的差异基因等
def fun1(adata):
    adata = adata[~adata.obs['gene'].isin(['None', 'CTRL'])]
    tmp = [i.split(',') for i in adata.obs['gene']]
    tmp = [i for i in tmp if i != 'CTRL' and i != 'None']
    tmp = np.unique([i for j in tmp for i in j])
    return tmp


def fun2(adata):
    mylist = []
    for gene, cell in zip(adata.obs['gene'], adata.obs_names):
        if gene == 'CTRL':
            mylist.append(cell)
        else:
            genes = gene.split(',')
            tmp = [True if i in adata.var_names else False for i in genes]
            if np.all(tmp): mylist.append(cell)
    adata1 = adata[mylist, :]
    return adata1

def fun3(x):   ### 只保留最多两个组合扰动
    xs = x.split(',')
    if len(xs) >= 3: return False
    else: return True


def fun4(adata):
    with open('/home/project/GW_PerturbSeq/geneEmbedding/geneList.pkl', 'rb') as fin:
        geneList = pickle.load(fin)
    tmp = adata.var_names.isin(geneList)  ### 必须有geneEmbedding
    adata = adata[:, tmp]
    adata1 = fun2(adata)   ### 扰动基因必须在表达谱中 且有geneEmbedding
    adata1.write_h5ad('raw.h5ad')


### 首先进行处理，使数据能够满足标准化处理, 全基因组文章三个数据集的处理方法
def QCsample1(dirName, fileName='Raw.h5ad'):
    os.chdir(dirName)
    adata = sc.read_h5ad(fileName)
    adata = adata[:, ~adata.var['gene_name'].duplicated()]  ## 更换index, 首先去除重复, 要不然后续报错
    adata.var['gene_id'] = adata.var.index
    adata.var.set_index('gene_name', inplace=True)
    adata.obs['gene'].replace({'non-targeting': 'CTRL'}, inplace=True)
    fun4(adata)

'''  确保都有embedding 并且 扰动基因在表达的基因中
adata = sc.read_h5ad('raw.h5ad')
[i for i in adata.obs['gene'].unique() if i not in geneList]
[i for i in adata.var_names if i not in geneList]
[i for i in adata.obs['gene'].unique() if i not in adata.var_names]
'''

####  张峰转录因子数据
def QCsample2(dirName):
    os.chdir(dirName)
    adata = sc.read_h5ad('Raw.h5ad')
    adata.var['gene_id'] = adata.var.index
    adata.var['gene_name'] = adata.var.index
    adata.var.set_index('gene_name', inplace=True)
    fun4(adata)

#### Perturb-CITE-seq
def QCsample3(dirName):
    os.chdir(dirName)
    adata = sc.read_h5ad("raw1.h5ad")
    adata.var['gene_id'] = adata.var.index
    adata.var['gene_name'] = adata.var.index
    adata.var.set_index('gene_name', inplace=True)
    tmp = adata.obs['gene'].apply(lambda x: fun3(x))   ### 只保留两个扰动
    adata = adata[tmp]
    fun4(adata)

### 对数据集进行差异基因的计算
###数据预处理
def f_preData(dirName):
    os.chdir(dirName)
    adata1 = sc.read_h5ad('raw.h5ad')

    filterNoneNums, filterCells, filterMT, filterMinNums, adata = preData(adata1, filterNone=True, minNums = 30, shuffle=False, filterCom=False,  seed = 42, mtpercent = 10,  min_genes = 200, domaxNums=500, doNor = True)
    print (filterNoneNums, filterCells, filterMT, filterMinNums)

    sc.pp.highly_variable_genes(adata, subset=False, n_top_genes=5000)  ###和gears一致，保证5000个hvg
    hvgs = list(adata.var_names[adata.var['highly_variable']])
    trainGene = fun1(adata)   ### 获得扰动的基因列表, 除去CTRL
    trainGene = [i for i in trainGene if i in adata.var_names]  ### 扰动数据必须保留在表达谱中
    keepGene = list(set(trainGene + hvgs))
    adata = adata[:, keepGene]
    adata1 = adata1[adata.obs_names, keepGene]
    
    adata = fun2(adata)   ### 再次保证扰动都在表达基因中, 不在的话过滤掉
    adata1 = fun2(adata1) ### 再次保证扰动都在表达基因中, 不在的话过滤掉
    adata.write_h5ad('filterNor.h5ad')
    adata1.write_h5ad('filterRaw.h5ad')




dirNames = [
    #'/home//project/GW_PerturbSeq/anndata/K562_GW',
    '/home//project/GW_PerturbSeq/anndata/K562_GW_subset',
    '/home//project/GW_PerturbSeq/anndata/K562_essential',
    '/home//project/GW_PerturbSeq/anndata/RPE1_essential',
    '/home//project/GW_PerturbSeq/anndata/TFatlas',

    '/home//project/GW_PerturbSeq/anndata_combination/PRJNA551220',
    '/home//project/GW_PerturbSeq/anndata_combination/Perturb-CITE-seq',
    '/home//project/GW_PerturbSeq/anndata_combination/PRJNA787633'
]

#dirName = '/home//project/GW_PerturbSeq/anndata/K562_GW_subset'; domaxNums = True; minNums = 30; ###QCsample1(dirName)   ###   ###  CRISPRi ### 每个扰动最多只保留50个细胞
dirName = '/home//project/GW_PerturbSeq/anndata/K562_GW'; domaxNums = True; minNums = 30; ##QCsample1(dirName)   ###   ###  CRISPRi
#dirName = '/home//project/GW_PerturbSeq/anndata/K562_essential'; domaxNums = True; minNums = 30; #QCsample1(dirName)
#dirName = '/home//project/GW_PerturbSeq/anndata/RPE1_essential'; domaxNums = True; minNums = 30; #QCsample1(dirName)
#dirName = '/home//project/GW_PerturbSeq/anndata/TFatlas'; domaxNums = True; minNums = 30; #QCsample2(dirName)   #Embryonic stem cells    Activation

#dirName = '/home//project/GW_PerturbSeq/anndata_combination/PRJNA551220'; domaxNums = True; minNums = 30; #QCsample3(dirName)   ### 组合扰动数据   ####   K562   Activation
#dirName = '/home//project/GW_PerturbSeq/anndata_combination/Perturb-CITE-seq'; domaxNums = True; minNums = 5; #QCsample3(dirName)  ####  CRISPRKO   melanoma cell
#dirName = '/home//project/GW_PerturbSeq/anndata_combination/PRJNA787633'; domaxNums = True; minNums = 5; #QCsample3(dirName)  ###  T cell    CRISPRa

### Ttest, wilcoxon, edgeR

if __name__ == '__main__':
    print ('hello, world')
    f_preData(dirName)
    getDEG("/home//project/GW_PerturbSeq/anndata/K562_GW", method='wilcoxon')
    
    for dirName in dirNames:
        for method in tqdm(['Ttest', 'wilcoxon', 'edgeR']):
            f_getDEG1(dirName, topDeg=0, method=method); f_getDEG1(dirName, topDeg=20, method=method)
            f_getDEG2(dirName, topDeg=0, method=method); f_getDEG2(dirName, topDeg=20, method=method)
            f_getDEG3(dirName, topDeg=0, method=method); f_getDEG3(dirName, topDeg=20, method=method)
            mergeLevelData(dirName, topDeg=0, method=method); mergeLevelData(dirName, topDeg=20, method=method)