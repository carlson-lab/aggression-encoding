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
from nmf_elastic import NMF_logistic


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8192)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

sys.path.append('/home/austin/DataAnalysis')
from data_tools import load_data

fnm='/media/austin/ThickBoy__1/DataAgression_Granger2/Aggression_sub_12.mat'
power,coherence,granger,labels = load_data(fnm,fBounds=(1,56),
                        feature_list=['power','coherence','granger'])

pf = labels['powerFeatures']
idxs_p = np.zeros(len(pf))
for i in range(len(pf)):
    if pf[i][:2] == 'PL':
        idxs_p[i] = 1
power = power[:,idxs_p==1]

myLabel = labels['windows']
mouse = np.asarray(myLabel['mouse'])
group = np.asarray(myLabel['group'])
expDate = np.asarray(myLabel['expDate'])
behavior = np.asarray(myLabel['behavior'])
behaviornon1 = np.asarray(myLabel['behaviornon1'])
time = np.asarray(myLabel['time'])
condition = np.asarray(myLabel['condition'])
N = len(mouse)

indx_pos = (behaviornon1==1)&(condition==4)
indx_neg1 = (behaviornon1==2)&(condition==4)
indx_neg2 = (behaviornon1==2)&(condition==6)
indx_neg3 = (behaviornon1==2)&(condition==8)
indx_neg = indx_neg1|indx_neg2|indx_neg3
indx_tot = indx_neg|indx_pos

y = np.zeros(N)
y[indx_pos] = 1

mouse = mouse[indx_tot]
group = group[indx_tot]
expDate = expDate[indx_tot]
behavior = behavior[indx_tot]
behaviornon1 = behaviornon1[indx_tot]
time = time[indx_tot]
condition = condition[indx_tot]
y = y[indx_tot]

N = len(mouse)

training_set_idx = np.ones(N)
training_set_idx[mouse=='Mouse048'] = 0
training_set_idx[mouse=='Mouse7980'] = 0
training_set_idx[mouse=='Mouse7998'] = 0

granger = np.exp(granger)
granger[granger>10] = 10
power = power*10
power[power>6] = 6

X = power
X = X[indx_tot]

print(mouse.shape)
print(X.shape)

X_train = X[training_set_idx==1]
m_train = mouse[training_set_idx==1]
y_train = y[training_set_idx==1]

X_test = X[training_set_idx==0]
m_test = mouse[training_set_idx==0]
y_test = y[training_set_idx==0]

mu = float(sys.argv[1])

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Add the newer data
power,coherence,granger,labels_new = load_data('/media/austin/ThickBoy__1/DataAgression_Granger2/CL_baseline_all_validate3.mat',fBounds=(1,56),feature_list=['power','coherence','granger'])

power = 10*power
power = power.astype(np.float32)
power[power>6] = 6
power = power[:,idxs_p==1]

coherence = coherence.astype(np.float32)
granger = np.exp(granger)
granger[granger>10] = 10
granger = granger.astype(np.float32)

X_new = power

windows_new = labels_new['windows']
mouse_new = np.squeeze(windows_new['mouse'])
expDate_new = np.squeeze(windows_new['expDate'])
group_new = np.squeeze(windows_new['group'])
condition_new = np.squeeze(windows_new['condition'])
behavior_new = np.squeeze(windows_new['behavior'])
time_new = np.squeeze(windows_new['time'])

idx_pos_new = (condition_new==4)&(behavior_new==1)
indx_neg_new = (behavior_new==2)&((condition_new==4)|(condition_new==6)|(condition_new==8))
y_new = np.zeros(len(mouse_new))
y_new[idx_pos_new] = 1
idx_tot_new = idx_pos_new|indx_neg_new

X_new = X_new[idx_tot_new]
mouse_new = mouse_new[idx_tot_new]
y_new = y_new[idx_tot_new]


mice_new = np.unique(mouse_new)
nMice = len(mice_new)
mice_new_train = mice_new[:4]


ids = np.zeros(len(mouse_new))
for i in range(4):
	ids[mouse_new==mice_new_train[i]] = 1

print('>>>>>>>>>>>>>.')
print(y_new.shape)
print(ids.shape)
print(mouse_new.shape)
print(X_new.shape)

X_train_new = X_new[ids==1,:]
X_test_new = X_new[ids==0,:]
y_train_new = y_new[ids==1]
y_test_new = y_new[ids==0]

print(X_train_new.shape)
print(y_train_new.shape)
print(X_test_new.shape)
print(y_test_new.shape)

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

X_train_tot = np.vstack((X_train,X_train_new))
y_train_tot = np.concatenate((y_train,y_train_new))
weights_g = np.ones(X_train_tot.shape[0])
weights_s = np.ones(X_train_tot.shape[0])

nFact = 8
myDict = {}

nIter = 20000
model = NMF_logistic(nFact,nIter=nIter,LR=1e-3,mu=mu,batchSize=100)
S_train = model.fit_transform(X_train_tot,y_train_tot)#,weights_s=weights_s,weights_g=weights_g)
S_train = model.transform(X_train)#,encoder)
S_test = model.transform(X_test)#,encoder)
S_train_new = model.transform(X_train_new)#,encoder)
S_test_new = model.transform(X_test_new)#,encoder)
components_ = model.components_

#sname = './checkpoints/Elastic_log_' + str(int(100*mu))
#encoder.save_weights(sname)

myDict['S_train'] = S_train
myDict['S_test'] = S_test
myDict['S_train_new'] = S_train_new
myDict['S_test_new'] = S_test_new
myDict['components'] = components_
myDict['phi'] = model.Phi

Ex = np.mean(X_train,axis=0)
reconRand_tr = np.mean((X_train-Ex)**2)
reconRand_te = np.mean((X_test-Ex)**2)
print('Random Reconstruction',np.mean((X_train-Ex)**2))
print('Random Reconstruction',np.mean((X_test-Ex)**2))
myDict['recon_random_train'] = reconRand_tr
myDict['recon_random_test'] = reconRand_te
myDict['Ex'] = Ex

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

myDict['A_enc'] = model.A_enc
myDict['B_enc'] = model.B_enc

mice_new = np.unique(mouse_new)
nMice = len(mice_new)
mice_new_train = mice_new[:4]

ids = np.zeros(len(mouse_new))
for i in range(4):
    ids[mouse_new==mice_new_train[i]] = 1
X_train_new = X_new[ids==1,:]
X_test_new = X_new[ids==0,:]
y_train_new = y_new[ids==1]
y_test_new = y_new[ids==0]

m_test_new = mouse_new[ids==0]

mice_test = np.unique(m_test_new)
for i in range(4):
    idxs = m_test_new==mice_test[i]
    myDict[mice_test[i]+'_auc'] = roc_auc_score(y_test_new[idxs],S_test_new[idxs,0])
    print(mice_test[i],myDict[mice_test[i]+'_auc'])

pname = 'NMF_PrLPowerOnly2' + str(mu) + '.p'
pickle.dump(myDict,open(pname,'wb'))



