import numpy as np
import sklearn.decomposition as dp
import pickle
import sys,os
import numpy.random as rand
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import auc,roc_curve,roc_auc_score
from sklearn.metrics import average_precision_score,precision_recall_curve
from sklearn.utils.random import sample_without_replacement
import tensorflow as tf

sys.path.append('/home/austin/Aggression/Code/NMF')
from nmf_joint2 import NMF_logistic

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

myDict = pickle.load(open('/home/austin/Aggression/Experiments/LimitedData.p','rb'))
X_train = myDict['X_train']
X_test = myDict['X_test']
y_train = myDict['y_train']
y_test = myDict['y_test']
m_train = myDict['mouse_train']
m_test = myDict['mouse_test']

weights_g = np.ones(X_train.shape[0])
weights_s = np.ones(X_train.shape[0])

mu = float(sys.argv[1])

nFact = 8
myDict = {}

nIter = 20000
model = NMF_logistic(nFact,nIter=nIter,LR=1e-3,mu=mu,batchSize=100)
S_train = model.fit_transform(X_train,y_train,weights_s=weights_s,
							weights_g=weights_g)
S_test = model.transform(X_test)
components_ = model.components_

myDict['S_train'] = S_train
myDict['S_test'] = S_test
myDict['components'] = components_
myDict['Phi_1'] = model.Phi1
myDict['Phi_2'] = model.Phi2

Ex = np.mean(X_train,axis=0)
reconRand_tr = np.mean((X_train-Ex)**2)
reconRand_te = np.mean((X_test-Ex)**2)
print('Random Reconstruction',np.mean((X_train-Ex)**2))
print('Random Reconstruction',np.mean((X_test-Ex)**2))
myDict['recon_random_train'] = reconRand_tr
myDict['recon_random_test'] = reconRand_te

X_recon_tr = np.dot(S_train,components_)
X_recon_te = np.dot(S_test,components_)
print('Recon sNMF',np.mean((X_train-X_recon_tr)**2))
print('Recon sNMF',np.mean((X_test-X_recon_te)**2))
recon_snmf_tr = np.mean((X_recon_tr-X_train)**2)
recon_snmf_te = np.mean((X_recon_te-X_test)**2)

myDict['recon_sae_train'] = recon_snmf_tr
myDict['recon_sae_test'] = recon_snmf_te

mice_test = np.unique(m_test)

print(model.Phi1,model.Phi2)

#sign = np.squeeze(np.sign(phi))
Str0 = S_train[:,0]*-1
Ste0 = S_test[:,0]*-1

Str1 = S_train[:,1]
Ste1 = S_test[:,1]

print('First components')
print('Training ROC 0',roc_auc_score(y_train,Str0))
print('Testing ROC 0',roc_auc_score(y_test,Ste0))
print('>>>>>>>>>>>>>>>>')
print('Training ROC 1',roc_auc_score(y_train,Str1))
print('Testing ROC 1',roc_auc_score(y_test,Ste1))

myDict['roc_train0'] = roc_auc_score(y_train,Str0)
myDict['roc_test0'] = roc_auc_score(y_test,Ste0)

myDict['roc_train1'] = roc_auc_score(y_train,Str1)
myDict['roc_test1'] = roc_auc_score(y_test,Ste1)

idx1 = mice_test[0]==m_test
idx2 = mice_test[1]==m_test
idx3 = mice_test[2]==m_test
print('ROC1',roc_auc_score(y_test[idx1],Ste0[idx1]))
print('ROC2',roc_auc_score(y_test[idx2],Ste0[idx2]))
print('ROC3',roc_auc_score(y_test[idx3],Ste0[idx3]))
print('>>>>>>>>>>>>>>>>')
print('ROC1',roc_auc_score(y_test[idx1],Ste1[idx1]))
print('ROC2',roc_auc_score(y_test[idx2],Ste1[idx2]))
print('ROC3',roc_auc_score(y_test[idx3],Ste1[idx3]))

myDict['rTest1_0'] = roc_auc_score(y_test[idx1],Ste0[idx1])
myDict['rTest2_0'] = roc_auc_score(y_test[idx2],Ste0[idx2])
myDict['rTest3_0'] = roc_auc_score(y_test[idx3],Ste0[idx3])

myDict['rTest1_1'] = roc_auc_score(y_test[idx1],Ste1[idx1])
myDict['rTest2_1'] = roc_auc_score(y_test[idx2],Ste1[idx2])
myDict['rTest3_1'] = roc_auc_score(y_test[idx3],Ste1[idx3])
#Now do precision/recall
print('>>>>>>>>>>>>>>>>>>')

myDict['pr_train'] = average_precision_score(y_train,Str0)
myDict['pr_test'] = average_precision_score(y_test,Ste0)

myDict['pr_test1_0'] = average_precision_score(y_test[idx1],Ste0[idx1])
myDict['pr_test2_0'] = average_precision_score(y_test[idx2],Ste0[idx2])
myDict['pr_test3_0'] = average_precision_score(y_test[idx3],Ste0[idx3])

myDict['pr_test1_1'] = average_precision_score(y_test[idx1],Ste1[idx1])
myDict['pr_test2_1'] = average_precision_score(y_test[idx2],Ste1[idx2])
myDict['pr_test3_1'] = average_precision_score(y_test[idx3],Ste1[idx3])

myDict['pr_random1'] = np.sum(y_test[idx1])/len(y_test[idx1])
myDict['pr_random2'] = np.sum(y_test[idx2])/len(y_test[idx2])
myDict['pr_random3'] = np.sum(y_test[idx3])/len(y_test[idx3])

precision1,recall1,_ = precision_recall_curve(y_test[idx1],Ste0[idx1])
precision2,recall2,_ = precision_recall_curve(y_test[idx2],Ste0[idx2])
precision3,recall3,_ = precision_recall_curve(y_test[idx3],Ste0[idx3])

myDict['precision1_0'] = precision1
myDict['precision2_0'] = precision2
myDict['precision3_0'] = precision3
myDict['recall1_0'] =recall1 
myDict['recall2_0'] =recall2 
myDict['recall3_0'] =recall3 

precision1,recall1,_ = precision_recall_curve(y_test[idx1],Ste1[idx1])
precision2,recall2,_ = precision_recall_curve(y_test[idx2],Ste1[idx2])
precision3,recall3,_ = precision_recall_curve(y_test[idx3],Ste1[idx3])

myDict['precision1_1'] = precision1
myDict['precision2_1'] = precision2
myDict['precision3_1'] = precision3
myDict['recall1_1'] =recall1 
myDict['recall2_1'] =recall2 
myDict['recall3_1'] =recall3 


pname = 'Unbalanced_Joint2_' + str(int(100*mu)) + '.p'
pickle.dump(myDict,open(pname,'wb'))
































