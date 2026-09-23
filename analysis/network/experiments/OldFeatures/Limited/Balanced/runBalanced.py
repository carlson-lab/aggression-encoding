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

sys.path.append('/home/austin/Aggression/Code/Balanced')
from nmf_supervised import NMF_logistic

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
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

#####################################
###                               ###
### Now read in the original data ###
###                               ###
#####################################
baseDir = '/media/austin/ThickBoy__1/DataAgression/'
pname = baseDir + 'Aggression_power.p'
cname1 = baseDir + 'Aggression_coherence1.p'
cname2 = baseDir + 'Aggression_coherence2.p'
cname3 = baseDir + 'Aggression_coherence3.p'
gname1 = baseDir + 'Aggression_granger1.p'
gname2 = baseDir + 'Aggression_granger2.p'
gname3 = baseDir + 'Aggression_granger3.p'

lname = baseDir + 'Agression_labels.p'
sname = baseDir + 'Agression_split_labels.p'

myP = pickle.load(open(pname,'rb'))
myC1 = pickle.load(open(cname1,'rb'))
myC2 = pickle.load(open(cname2,'rb'))
myC3 = pickle.load(open(cname3,'rb'))
myG1 = pickle.load(open(gname1,'rb'))
myG2 = pickle.load(open(gname2,'rb'))
myG3 = pickle.load(open(gname3,'rb'))

mySplits = pickle.load(open(sname,'rb'))
myLabels = pickle.load(open(lname,'rb'))

# Load all the labels
mouse_idx,mice = myLabels['mouse_idx'],myLabels['mice']
expD_idx,expDates= myLabels['epxD_idx'],myLabels['expDates']
group_idx,groups= myLabels['group_idx'],myLabels['groups']
condition_idx,conditions= myLabels['condition_idx'],myLabels['conditions']
behavior_idx,behaviors= myLabels['behavior_idx'],myLabels['behaviors']

power = myP['power']*10
power = power.astype(np.float32)
print(np.mean(power>6))
power[power>6] = 6

C1 = myC1['coherence']
C2 = myC2['coherence']
C3 = myC3['coherence']
coherence = np.vstack((C1,C2,C3))
coherence = coherence.astype(np.float32)

G1 = myG1['granger']
G2 = myG2['granger']
G3 = myG3['granger']
granger= np.vstack((G1,G2,G3))
granger = np.exp(granger)
granger[granger>10] = 10
granger = granger.astype(np.float32)

X_tot = np.hstack((power,coherence,granger))

N = len(mouse_idx)
training_set_idx = np.ones(N)
training_set_idx[mouse_idx==mice.index('Mouse048')] = 0
training_set_idx[mouse_idx==mice.index('Mouse7980')] = 0
training_set_idx[mouse_idx==mice.index('Mouse7998')] = 0

X_tot_sub = X_tot[training_set_idx==1]
print(X_tot_sub.shape)
## This is our complete unsupervised dataset


#### Continue as normal

mu = float(sys.argv[1])

nFact = 8
myDict = {}

nIter = 5000
model = NMF_logistic(nFact,nIter=nIter,LR=1e-3,mu=mu,batchSize=100)
S_train = model.fit_transform(X_tot_sub,X_train,y_train)
S_test = model.transform(X_test)
components_ = model.components_

myDict['losses_gen'] = model.losses_gen
myDict['losses_sup'] = model.losses_sup
myDict['losses_rec'] = model.losses_recon

myDict['S_train'] = S_train
myDict['S_test'] = S_test
myDict['components'] = components_
myDict['phi'] = model.Phi

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

phi = model.Phi
print(phi)

sign = np.squeeze(np.sign(phi))
Str0 = S_train[:,0]*sign
Ste0 = S_test[:,0]*sign

print('First components')
print('Training ROC',roc_auc_score(y_train,Str0))
print('Testing ROC',roc_auc_score(y_test,Ste0))
myDict['roc_train'] = roc_auc_score(y_train,Str0)
myDict['roc_test'] = roc_auc_score(y_test,Ste0)

idx1 = mice_test[0]==m_test
idx2 = mice_test[1]==m_test
idx3 = mice_test[2]==m_test
print('ROC1',roc_auc_score(y_test[idx1],Ste0[idx1]))
print('ROC2',roc_auc_score(y_test[idx2],Ste0[idx2]))
print('ROC3',roc_auc_score(y_test[idx3],Ste0[idx3]))

myDict['rTest1'] = roc_auc_score(y_test[idx1],Ste0[idx1])
myDict['rTest2'] = roc_auc_score(y_test[idx2],Ste0[idx2])
myDict['rTest3'] = roc_auc_score(y_test[idx3],Ste0[idx3])

#Now do precision/recall
print('>>>>>>>>>>>>>>>>>>')

myDict['pr_train'] = average_precision_score(y_train,Str0)
myDict['pr_test'] = average_precision_score(y_test,Ste0)

myDict['pr_test1'] = average_precision_score(y_test[idx1],Ste0[idx1])
myDict['pr_test2'] = average_precision_score(y_test[idx2],Ste0[idx2])
myDict['pr_test3'] = average_precision_score(y_test[idx3],Ste0[idx3])

myDict['pr_random1'] = np.sum(y_test[idx1])/len(y_test[idx1])
myDict['pr_random2'] = np.sum(y_test[idx2])/len(y_test[idx2])
myDict['pr_random3'] = np.sum(y_test[idx3])/len(y_test[idx3])

precision1,recall1,_ = precision_recall_curve(y_test[idx1],Ste0[idx1])
precision2,recall2,_ = precision_recall_curve(y_test[idx2],Ste0[idx2])
precision3,recall3,_ = precision_recall_curve(y_test[idx3],Ste0[idx3])

myDict['precision1'] = precision1
myDict['precision2'] = precision2
myDict['precision3'] = precision3
myDict['recall1'] =recall1 
myDict['recall2'] =recall2 
myDict['recall3'] =recall3 

pname = 'Balanced_' + str(int(100*mu)) + '.p'
pickle.dump(myDict,open(pname,'wb'))






























