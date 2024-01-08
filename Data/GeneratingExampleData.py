import numpy as np
import scanpy as sc
import anndata as ad
import joblib
import scipy
import os

np.random.seed(888)

# generating the training example data
train_data_template_x = (np.random.rand(1000,5000) > 0.5) * 1
train_data_layer_1 = (np.random.rand(1000,5000) > 0.5) * 1
train_data_layer_2 = (np.random.rand(1000,5000) > 0.5) * 1
train_data_layer_3 = np.random.rand(1000,5000)
train_data = ad.AnnData(X = scipy.sparse.csr_matrix(train_data_template_x))
train_data.layers['level1'] = scipy.sparse.csr_matrix(train_data_layer_1)
train_data.layers['level2'] = scipy.sparse.csr_matrix(train_data_layer_2)
train_data.layers['level3'] = scipy.sparse.csr_matrix(train_data_layer_3)
train_data.obs.index = [f"Gene{i+1}" for i in range(1000)]
train_data.var.index = [f"Gene{i+1}" for i in range(5000)]

# generating the validation example data
val_data_template_x = (np.random.rand(100,5000) > 0.5) * 1
val_data_layer_1 = (np.random.rand(100,5000) > 0.5) * 1
val_data_layer_2 = (np.random.rand(100,5000) > 0.5) * 1
val_data_layer_3 = np.random.rand(100,5000)
val_data = ad.AnnData(X = scipy.sparse.csr_matrix(val_data_template_x))
val_data.layers['level1'] = scipy.sparse.csr_matrix(val_data_layer_1)
val_data.layers['level2'] = scipy.sparse.csr_matrix(val_data_layer_2)
val_data.layers['level3'] = scipy.sparse.csr_matrix(val_data_layer_3)
val_data.obs.index = [f"Gene{i+1}" for i in range(1000,1100)]
val_data.var.index = [f"Gene{i+1}" for i in range(5000)]

# generating the testing example data
test_data_template_x = (np.random.rand(200,5000) > 0.5) * 1
test_data_layer_1 = (np.random.rand(200,5000) > 0.5) * 1
test_data_layer_2 = (np.random.rand(200,5000) > 0.5) * 1
test_data_layer_3 = np.random.rand(200,5000)
test_data = ad.AnnData(X = scipy.sparse.csr_matrix(test_data_template_x))
test_data.layers['level1'] = scipy.sparse.csr_matrix(test_data_layer_1)
test_data.layers['level2'] = scipy.sparse.csr_matrix(test_data_layer_2)
test_data.layers['level3'] = scipy.sparse.csr_matrix(test_data_layer_3)
test_data.obs.index = [f"Gene{i+1},Gene{i+2}" for i in range(1100,1300)]
test_data.var.index = [f"Gene{i+1}" for i in range(5000)]

# generating the top 40 DEGs for testing example data
test_data_top40 = ad.AnnData(X = scipy.sparse.csr_matrix(test_data_template_x))
test_data_layer_1_top40 = np.zeros_like(np.random.rand(200,5000))
for i in range(200):
    test_data_layer_1_top40[i][np.random.choice(5000,40,replace=False)]=1
test_data_top40.layers['level1'] = scipy.sparse.csr_matrix(test_data_layer_1_top40)
test_data_top40.layers['level2'] = scipy.sparse.csr_matrix(test_data_layer_2)
test_data_top40.layers['level3'] = scipy.sparse.csr_matrix(test_data_layer_3)
test_data_top40.obs.index = [f"Gene{i+1},Gene{i+2}" for i in range(1100,1300)]
test_data_top40.var.index = [f"Gene{i+1}" for i in range(5000)]

# generating the gene embedding matrix and the gene embedding orders must be consistent with the gene orders of data.var
gene_embs = np.random.rand(5000, 512).astype('float32')

if not os.path.exists("./Data"):
    os.makedirs("./Data")

train_data.write("./Data/example_train.h5ad")
val_data.write("./Data/example_val.h5ad")
test_data.write("./Data/example_test.h5ad")
test_data_top40.write("./Data/example_test_top40.h5ad")
joblib.dump(gene_embs, "./Data/example_gene_embeddings.pkl")