import numpy as np
import sklearn.decomposition as dp
import pickle
import sys,os
import numpy.random as rand
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import auc,roc_curve,roc_auc_score
from sklearn.metrics import average_precision_score,precision_recall_curve
from sklearn.utils.random import sample_without_replacement

sys.path.append('/home/austin/DataAnalysis')
from data_tools import load_data

fnm='/media/austin/ThickBoy__1/DataAgression_Granger2/Aggression_sub_12.mat'
power,labels = load_data(fnm,fBounds=(1,56),
                        feature_list=['power'])

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

m_train = mouse[training_set_idx==1]
y_train = y[training_set_idx==1]

m_test = mouse[training_set_idx==0]
y_test = y[training_set_idx==0]

mu = float(sys.argv[1])

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Add the newer data
power,labels_new = load_data('/media/austin/ThickBoy__1/DataAgression_Granger2/CL_baseline_all_validate3.mat',fBounds=(1,56),feature_list=['power'])


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

mouse_new = mouse_new[idx_tot_new]
y_new = y_new[idx_tot_new]

mice_new = np.unique(mouse_new)
nMice = len(mice_new)
mice_new_train = mice_new[:4]


ids = np.zeros(len(mouse_new))
for i in range(4):
	ids[mouse_new==mice_new_train[i]] = 1

idx_mt_new5 = (mice_new[5]==mouse_new)
idx_mt_new6 = (mice_new[6]==mouse_new)
idx_mt_new7 = (mice_new[7]==mouse_new)
idx_mt_new8 = (mice_new[8]==mouse_new)

print('>>>>>>>>>>>>>.')
print(y_new.shape)
print(ids.shape)
print(mouse_new.shape)

y_train_new = y_new[ids==1]
y_test_new = y_new[ids==0]

print(X_train_new.shape)
print(y_train_new.shape)
print(X_test_new.shape)
print(y_test_new.shape)

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

y_train_tot = np.concatenate((y_train,y_train_new))

myDict = pickle.load(open('Unbalanced_Elastic_12_enc_1.0.p','rb'))

S_train = myDict['S_train'] 
S_test = myDict['S_test'] 
S_train_new = myDict['S_train_new'] 
S_test_new = myDict['S_test_new'] 

S_tr = np.vstack((S_train,S_train_new))

model_lr = LR()
model_lr.fit(S_tr,y_train_tot)

mice_test = np.unique(m_test)

#sign = np.squeeze(np.sign(phi))
Str0 = S_train[:,0]*-1
Ste0 = S_test[:,0]*-1

Str6 = S_train[:,5]
Ste6 = S_test[:,5]

Scomb = model_lr.decision_function(S_test)
Scomb_new = model_lr.decision_function(S_test_new)

myDict2 = {}

myDict2['roc_train0'] = roc_auc_score(y_train,Str0)
myDict2['roc_test0'] = roc_auc_score(y_test,Ste0)

myDict2['roc_train6'] = roc_auc_score(y_train,Str6)
myDict2['roc_test6'] = roc_auc_score(y_test,Ste6)

idx1 = mice_test[0]==m_test
idx2 = mice_test[1]==m_test
idx3 = mice_test[2]==m_test
print('ROC1',roc_auc_score(y_test[idx1],Ste0[idx1]))
print('ROC2',roc_auc_score(y_test[idx2],Ste0[idx2]))
print('ROC3',roc_auc_score(y_test[idx3],Ste0[idx3]))
print('>>>>>>>>>>>>>>>>')
print('ROC1',roc_auc_score(y_test[idx1],Ste6[idx1]))
print('ROC2',roc_auc_score(y_test[idx2],Ste6[idx2]))
print('ROC3',roc_auc_score(y_test[idx3],Ste6[idx3]))
print('>>>>>>>>>>>>>>>>')

myDict2['rTest1_0'] = roc_auc_score(y_test[idx1],Ste0[idx1])
myDict2['rTest2_0'] = roc_auc_score(y_test[idx2],Ste0[idx2])
myDict2['rTest3_0'] = roc_auc_score(y_test[idx3],Ste0[idx3])

myDict2['rTest1_6'] = roc_auc_score(y_test[idx1],Ste6[idx1])
myDict2['rTest2_6'] = roc_auc_score(y_test[idx2],Ste6[idx2])
myDict2['rTest3_6'] = roc_auc_score(y_test[idx3],Ste6[idx3])

pname = 'Combined_preds.p'
pickle.dump(myDict,open(pname,'wb'))




























