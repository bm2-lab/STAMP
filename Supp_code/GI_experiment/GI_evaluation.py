import joblib
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from scipy import stats
import pandas as pd
np.random.seed(000)
GIs = {
    'NEOMORPHIC': ['CBL+TGFBR2',
                  'KLF1+TGFBR2',
                  'MAP2K6+SPI1',
                  'SAMD1+TGFBR2',
                  'TGFBR2+C19orf26',
                  'TGFBR2+ETS2',
                  'CBL+UBASH3A',
                  'CEBPE+KLF1',
                  'DUSP9+MAPK1',
                  'FOSB+PTPN12',
                  'PLK4+STIL',
                  'PTPN12+OSR2',
                  'ZC3HAV1+CEBPE'],
    'ADDITIVE': ['BPGM+SAMD1',
                'CEBPB+MAPK1',
                'CEBPB+OSR2',
                'DUSP9+PRTG',
                'FOSB+OSR2',
                'IRF1+SET',
                'MAP2K3+ELMSAN1',
                'MAP2K6+ELMSAN1',
                'POU3F2+FOXL2',
                'RHOXF2BB+SET',
                'SAMD1+PTPN12',
                'SAMD1+UBASH3B',
                'SAMD1+ZBTB1',
                'SGK1+TBX2',
                'TBX3+TBX2',
                'ZBTB10+SNAI1'],
    'EPISTASIS': ['AHR+KLF1',
                 'MAPK1+TGFBR2',
                 'TGFBR2+IGDCC3',
                 'TGFBR2+PRTG',
                 'UBASH3B+OSR2',
                 'DUSP9+ETS2',
                 'KLF1+CEBPA',
                 'MAP2K6+IKZF3',
                 'ZC3HAV1+CEBPA'],
    'REDUNDANT': ['CDKN1C+CDKN1A',
                 'MAP2K3+MAP2K6',
                 'CEBPB+CEBPA',
                 'CEBPE+CEBPA',
                 'CEBPE+SPI1',
                 'ETS2+MAPK1',
                 'FOSB+CEBPE',
                 'FOXA3+FOXA1'],
    'POTENTIATION': ['CNN1+UBASH3A',
                    'ETS2+MAP7D1',
                    'FEV+CBFA2T3',
                    'FEV+ISL2',
                    'FEV+MAP7D1',
                    'PTPN12+UBASH3A'],
    'SYNERGY_SIMILAR_PHENO':['CBL+CNN1',
                            'CBL+PTPN12',
                            'CBL+PTPN9',
                            'CBL+UBASH3B',
                            'FOXA3+FOXL2',
                            'FOXA3+HOXB9',
                            'FOXL2+HOXB9',
                            'UBASH3B+CNN1',
                            'UBASH3B+PTPN12',
                            'UBASH3B+PTPN9',
                            'UBASH3B+ZBTB25'],
    'SYNERGY_DISSIMILAR_PHENO': ['AHR+FEV',
                                'DUSP9+SNAI1',
                                'FOXA1+FOXF1',
                                'FOXA1+FOXL2',
                                'FOXA1+HOXB9',
                                'FOXF1+FOXL2',
                                'FOXF1+HOXB9',
                                'FOXL2+MEIS1',
                                'IGDCC3+ZBTB25',
                                'POU3F2+CBFA2T3',
                                'PTPN12+ZBTB25',
                                'SNAI1+DLX2',
                                'SNAI1+UBASH3B'],
    'SUPPRESSOR': ['CEBPB+PTPN12',
                  'CEBPE+CNN1',
                  'CEBPE+PTPN12',
                  'CNN1+MAPK1',
                  'ETS2+CNN1',
                  'ETS2+IGDCC3',
                  'ETS2+PRTG',
                  'FOSB+UBASH3B',
                  'IGDCC3+MAPK1',
                  'LYL1+CEBPB',
                  'MAPK1+PRTG',
                  'PTPN12+SNAI1']
}
GIs['SYNERGY'] = GIs['SYNERGY_DISSIMILAR_PHENO'] + GIs['SYNERGY_SIMILAR_PHENO'] + GIs['POTENTIATION']

all_results = joblib.load("./STAMP_results.pkl")
all_results_truth = joblib.load("./Truth_results.pkl")
all_results_gears = joblib.load("./Gears_results.pkl")
science_data = pd.read_csv("./Science_data.csv", sep = ',')
all_results_science = {}
for idx, name in enumerate(science_data['name']):
    all_results_science[(','.join(name.split('_')))] = {}
    all_results_science[(','.join(name.split('_')))]['magnitude'] = science_data['ts_norm2'][idx]
    all_results_science[(','.join(name.split('_')))]['model_fit'] = science_data['ts_linear_dcor'][idx]
    all_results_science[(','.join(name.split('_')))]['equality_of_contribution'] = science_data['dcor_ratio'][idx]
    all_results_science[(','.join(name.split('_')))]['Similarity'] = science_data['dcor'][idx]
    
    
def calculate_metric(all_results, top_k = 10):
    mags = [all_results[i]['magnitude'] for i in all_results]
    model_fits = [all_results[i]['model_fit'] for i in all_results]
    equality_of_contributions = [all_results[i]['equality_of_contribution'] for i in all_results]
    Similaritys = [all_results[i]['Similarity'] for i in all_results]
    top10_synergy = np.array(list(all_results.keys()))[np.argsort(mags)[::-1][:top_k]]
    top10_precision_synergy = len(set(top10_synergy).intersection(set([(',').join(i.split("+")) for i in GIs['SYNERGY']]).union(set([(',').join(i.split("+")[::-1]) for i in GIs['SYNERGY']]))))
    top10_precision_synergy /= top_k
    top10_suppressor = np.array(list(all_results.keys()))[np.argsort(mags)[:top_k]]
    top10_precision_suppressor = len(set(top10_suppressor).intersection(set([(',').join(i.split("+")) for i in GIs['SUPPRESSOR']]).union(set([(',').join(i.split("+")[::-1]) for i in GIs['SUPPRESSOR']]))))
    top10_precision_suppressor /= top_k
    top10_neomorphism = np.array(list(all_results.keys()))[np.argsort(model_fits)[:top_k]]
    top10_precision_neomorphism = len(set(top10_neomorphism).intersection(set([(',').join(i.split("+")) for i in GIs['NEOMORPHIC']]).union(set([(',').join(i.split("+")[::-1]) for i in GIs['NEOMORPHIC']]))))
    top10_precision_neomorphism /= top_k
    top10_redundant = np.array(list(all_results.keys()))[np.argsort(Similaritys)[::-1][:top_k]]
    top10_precision_redundant = len(set(top10_redundant).intersection(set([(',').join(i.split("+")) for i in GIs['REDUNDANT']]).union(set([(',').join(i.split("+")[::-1]) for i in GIs['REDUNDANT']]))))
    top10_precision_redundant /= 8
    top10_epistasis = np.array(list(all_results.keys()))[np.argsort(equality_of_contributions)[:top_k]]
    top10_precision_epistasis = len(set(top10_epistasis).intersection(set([(',').join(i.split("+")) for i in GIs['EPISTASIS']]).union(set([(',').join(i.split("+")[::-1]) for i in GIs['EPISTASIS']]))))
    top10_precision_epistasis /= 9
    return top10_precision_synergy, top10_precision_suppressor, top10_precision_neomorphism, top10_precision_redundant, top10_precision_epistasis

print("Science_ori,Top10 precision:",calculate_metric(all_results_science))
print("Ground_truth,Top10 precision:",calculate_metric(all_results_truth))
print("STAMP,Top10 precision:",calculate_metric(all_results))
print("GEARs,Top10 precision:",calculate_metric(all_results_gears))

def calculate_metric_top10acc(all_results,all_results_truth, top_k = 10):
    mags = [all_results[i]['magnitude'] for i in all_results]
    model_fits = [all_results[i]['model_fit'] for i in all_results]
    equality_of_contributions = [all_results[i]['equality_of_contribution'] for i in all_results]
    Similaritys = [all_results[i]['Similarity'] for i in all_results]
    mags_truth = [all_results_truth[i]['magnitude'] for i in all_results_truth]
    model_fits_truth = [all_results_truth[i]['model_fit'] for i in all_results_truth]
    equality_of_contributions_truth = [all_results_truth[i]['equality_of_contribution'] for i in all_results_truth]
    Similaritys_truth = [all_results_truth[i]['Similarity'] for i in all_results_truth]
    
    top10_synergy = np.array(list(all_results.keys()))[np.argsort(mags)[::-1][:top_k]]
    top10_acc_synergy = set(top10_synergy).intersection(set([(',').join(i.split("+")) for i in GIs['SYNERGY']]).union(set([(',').join(i.split("+")[::-1]) for i in GIs['SYNERGY']])))
    top10_synergy_truth = np.array(list(all_results_truth.keys()))[np.argsort(mags_truth)[::-1][:top_k]]
    top10_acc_synergy_truth = set(top10_synergy_truth).intersection(set([(',').join(i.split("+")) for i in GIs['SYNERGY']]).union(set([(',').join(i.split("+")[::-1]) for i in GIs['SYNERGY']])))
    top10_acc_synergy = len(top10_acc_synergy.intersection(top10_acc_synergy_truth))/len(top10_acc_synergy_truth)
    
    top10_suppressor = np.array(list(all_results.keys()))[np.argsort(mags)[:top_k]]
    top10_acc_suppressor = set(top10_suppressor).intersection(set([(',').join(i.split("+")) for i in GIs['SUPPRESSOR']]).union(set([(',').join(i.split("+")[::-1]) for i in GIs['SUPPRESSOR']])))
    top10_suppressor_truth = np.array(list(all_results_truth.keys()))[np.argsort(mags_truth)[:top_k]]
    top10_acc_suppressor_truth = set(top10_suppressor_truth).intersection(set([(',').join(i.split("+")) for i in GIs['SUPPRESSOR']]).union(set([(',').join(i.split("+")[::-1]) for i in GIs['SUPPRESSOR']])))
    try:
        top10_acc_suppressor = len(top10_acc_suppressor.intersection(top10_acc_suppressor_truth))/len(top10_acc_suppressor_truth)
    except:
        top10_acc_suppressor=0
    
    top10_neomorphism = np.array(list(all_results.keys()))[np.argsort(model_fits)[:top_k]]
    top10_acc_neomorphism = set(top10_neomorphism).intersection(set([(',').join(i.split("+")) for i in GIs['NEOMORPHIC']]).union(set([(',').join(i.split("+")[::-1]) for i in GIs['NEOMORPHIC']])))
    top10_neomorphism_truth = np.array(list(all_results_truth.keys()))[np.argsort(model_fits)[:top_k]]
    top10_acc_neomorphism_truth = set(top10_neomorphism_truth).intersection(set([(',').join(i.split("+")) for i in GIs['NEOMORPHIC']]).union(set([(',').join(i.split("+")[::-1]) for i in GIs['NEOMORPHIC']])))
    top10_acc_neomorphism = len(top10_acc_neomorphism.intersection(top10_acc_neomorphism_truth))/len(top10_acc_neomorphism_truth)
    
    top10_redundant = np.array(list(all_results.keys()))[np.argsort(Similaritys)[::-1][:top_k]]
    top10_acc_redundant = set(top10_redundant).intersection(set([(',').join(i.split("+")) for i in GIs['REDUNDANT']]).union(set([(',').join(i.split("+")[::-1]) for i in GIs['REDUNDANT']])))
    top10_redundant_truth = np.array(list(all_results_truth.keys()))[np.argsort(Similaritys)[::-1][:top_k]]
    top10_acc_redundant_truth = set(top10_redundant_truth).intersection(set([(',').join(i.split("+")) for i in GIs['REDUNDANT']]).union(set([(',').join(i.split("+")[::-1]) for i in GIs['REDUNDANT']])))
    try:
        top10_acc_redundant = len(top10_acc_redundant.intersection(top10_acc_redundant_truth))/len(top10_acc_redundant_truth)
    except:
        top10_acc_redundant=0
    
    top10_epistasis = np.array(list(all_results.keys()))[np.argsort(equality_of_contributions)[:top_k]]
    top10_acc_epistasis = set(top10_epistasis).intersection(set([(',').join(i.split("+")) for i in GIs['EPISTASIS']]).union(set([(',').join(i.split("+")[::-1]) for i in GIs['EPISTASIS']])))
    top10_epistasis_truth = np.array(list(all_results_truth.keys()))[np.argsort(equality_of_contributions)[:top_k]]
    top10_acc_epistasis_truth = set(top10_epistasis_truth).intersection(set([(',').join(i.split("+")) for i in GIs['EPISTASIS']]).union(set([(',').join(i.split("+")[::-1]) for i in GIs['EPISTASIS']])))
    top10_acc_epistasis = len(top10_acc_epistasis.intersection(top10_acc_epistasis_truth))/len(top10_acc_epistasis_truth)
    
    return top10_acc_synergy, top10_acc_suppressor, top10_acc_neomorphism, top10_acc_redundant, top10_acc_epistasis

# print("STAMP,Top10 acc:",calculate_metric_top10acc(all_results, all_results_truth))
# print("GEARs,Top10 acc:",calculate_metric_top10acc(all_results_gears, all_results_truth))

def array_generation(results):
    X = []
    Y = []
    for idx, combo_gene in enumerate(results):
        if ('+').join(combo_gene.split(",")) in GIs['SYNERGY'] or ('+').join(combo_gene.split(",")[::-1]) in GIs['SYNERGY']:
            Y.append(0)
        elif ('+').join(combo_gene.split(",")) in GIs['SUPPRESSOR'] or ('+').join(combo_gene.split(",")[::-1]) in GIs['SUPPRESSOR']:
            Y.append(1)
        elif ('+').join(combo_gene.split(",")) in GIs['NEOMORPHIC'] or ('+').join(combo_gene.split(",")[::-1]) in GIs['NEOMORPHIC']:
            Y.append(2)
        elif ('+').join(combo_gene.split(",")) in GIs['REDUNDANT'] or ('+').join(combo_gene.split(",")[::-1]) in GIs['REDUNDANT']:
            Y.append(3)
        elif ('+').join(combo_gene.split(",")) in GIs['EPISTASIS'] or ('+').join(combo_gene.split(",")[::-1]) in GIs['EPISTASIS']:
            Y.append(4)
        elif ('+').join(combo_gene.split(",")) in GIs['ADDITIVE'] or ('+').join(combo_gene.split(",")[::-1]) in GIs['ADDITIVE']:
            Y.append(5)
        else:
            continue
        X.append([results[combo_gene][feature] for feature in results[combo_gene]])
    return np.array(X), np.array(Y)     

def acc_cal(X_STAMP,Y_STAMP):
    tmp = clf.predict_proba(X_STAMP)
    acc_SYNERGY = (Y_STAMP[Y_STAMP==0]==(tmp.argmax(axis=1)[Y_STAMP==0])).mean()
    acc_SUPPRESSOR = (Y_STAMP[Y_STAMP==1]==(tmp.argmax(axis=1)[Y_STAMP==1])).mean()
    acc_NEOMORPHIC = (Y_STAMP[Y_STAMP==2]==(tmp.argmax(axis=1)[Y_STAMP==2])).mean()
    acc_REDUNDANT = (Y_STAMP[Y_STAMP==3]==(tmp.argmax(axis=1)[Y_STAMP==3])).mean()
    acc_EPISTASIS = (Y_STAMP[Y_STAMP==4]==(tmp.argmax(axis=1)[Y_STAMP==4])).mean()
    acc_Additive = (Y_STAMP[Y_STAMP==5]==(tmp.argmax(axis=1)[Y_STAMP==5])).mean()
    return (acc_SYNERGY,acc_SUPPRESSOR, acc_NEOMORPHIC, acc_REDUNDANT, acc_EPISTASIS, acc_Additive)

X_truth,Y_truth = array_generation(all_results_truth)

X_truth = (X_truth-X_truth.mean(axis=0))/(X_truth.std(axis=0))

clf = tree.DecisionTreeClassifier(random_state=42, max_depth=6,min_samples_leaf=8, min_samples_split=8, max_leaf_nodes=6)
clf = clf.fit(X_truth, Y_truth)
plt.figure(figsize=(12, 8))
plot_tree(clf, filled=True, feature_names=["magnitude", "model_fit", "equality_of_contribution", "Similarity"], class_names=["SYNERGY", "SUPPRESSOR", "NEOMORPHIC", "REDUNDANT", "EPISTASIS","Additive"])
plt.savefig("./test_tree.png")

print("Ground_truth, accuracy",acc_cal(X_truth, Y_truth))

X_STAMP,Y_STAMP = array_generation(all_results)
X_STAMP = (X_STAMP-X_STAMP.mean(axis=0))/(X_STAMP.std(axis=0))
print("STAMP, accuracy",acc_cal(X_STAMP, Y_STAMP))


X_GEARs,Y_GEARs = array_generation(all_results_gears)
X_GEARs = (X_GEARs-X_GEARs.mean(axis=0))/(X_GEARs.std(axis=0))
print("GEARs, accuracy",acc_cal(X_GEARs, Y_GEARs))

##### Random test
acc_SYNERGY = 0
acc_SUPPRESSOR = 0
acc_NEOMORPHIC = 0
acc_REDUNDANT = 0
acc_EPISTASIS = 0
acc_Additive = 0
for i in range(10):
    random_idx = list(range(len(Y_truth)))
    np.random.shuffle(random_idx)
    random_Y_truth = Y_truth[random_idx]
    acc_SYNERGY += (Y_truth[Y_truth==0]==(random_Y_truth[Y_truth==0])).mean()
    acc_SUPPRESSOR += (Y_truth[Y_truth==1]==(random_Y_truth[Y_truth==1])).mean()
    acc_NEOMORPHIC += (Y_truth[Y_truth==2]==(random_Y_truth[Y_truth==2])).mean()
    acc_REDUNDANT += (Y_truth[Y_truth==3]==(random_Y_truth[Y_truth==3])).mean()
    acc_EPISTASIS += (Y_truth[Y_truth==4]==(random_Y_truth[Y_truth==4])).mean()
    acc_Additive += (Y_truth[Y_truth==5]==(random_Y_truth[Y_truth==5])).mean()
acc_SYNERGY /= i+1;acc_SUPPRESSOR /= i+1;acc_NEOMORPHIC/=i+1;acc_REDUNDANT/=i+1;acc_EPISTASIS/=i+1;acc_Additive/=i+1
print("Random, accuracy",acc_SYNERGY,acc_SUPPRESSOR,acc_NEOMORPHIC,acc_REDUNDANT,acc_EPISTASIS,acc_Additive)

