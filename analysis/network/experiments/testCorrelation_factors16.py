import numpy as np
import pickle
from scipy.stats import spearmanr
myDict = pickle.load(open('Unbalanced_Elastic_12_enc_1.0.p','rb'))
Strain = myDict['S_train']
Stest = myDict['S_test']
Strain_new = myDict['S_train_new']
Stest_new = myDict['S_test_new']

Str = np.vstack((Strain,Strain_new))
Ste = np.vstack((Stest,Stest_new))
Sto = np.vstack((Str,Ste))

corr_tr,p_tr = spearmanr(Str[:,0],Str[:,5])
corr_te,p_te = spearmanr(Ste[:,0],Ste[:,5])
corr_to,p_to = spearmanr(Sto[:,0],Sto[:,5])

