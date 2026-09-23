import numpy as np
import sklearn.decomposition as dp
import pickle
import sys,os
import numpy.random as rand
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import auc,roc_curve,roc_auc_score
from sklearn.utils.random import sample_without_replacement
import tensorflow as tf

sys.path.append('/home/austin/Aggression/Code/NMF')
from nmf_supervised import NMF_logistic


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
behaviornon1_idx,behaviorsnon1s= myLabels['behaviornon1_idx'],myLabels['behaviorsnon1s']

#######################
##                   ##
## Generate features ##
##                   ##
#######################
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

X = np.hstack((power,coherence,granger))

###################################
###################################
##                               ##
##  Divide training and testing  ##
##                               ##
###################################
###################################

N = len(mouse_idx)
training_set_idx = np.ones(N)
training_set_idx[mouse_idx==mice.index('Mouse048')] = 0
training_set_idx[mouse_idx==mice.index('Mouse7980')] = 0
training_set_idx[mouse_idx==mice.index('Mouse7998')] = 0

# Divide the training and testing sets
X_train = X[training_set_idx==1]
X_test = X[training_set_idx==0]

mouse_idx_train = mouse_idx[training_set_idx==1]
mouse_idx_test = mouse_idx[training_set_idx==0]
print(np.unique(mouse_idx_test))
print(mouse_idx_test.shape)

expDate_idx_train = expD_idx[training_set_idx==1]
expDate_idx_test = expD_idx[training_set_idx==0]

group_idx_train = group_idx[training_set_idx==1]
group_idx_test = group_idx[training_set_idx==0]

condition_idx_train = condition_idx[training_set_idx==1]
condition_idx_test = condition_idx[training_set_idx==0]

behavior_idx_train = behavior_idx[training_set_idx==1]
behavior_idx_test = behavior_idx[training_set_idx==0]

aggression_idx_train = behaviornon1_idx[training_set_idx==1]
aggression_idx_test = behaviornon1_idx[training_set_idx==0]

#Numbers of observations in each set
N_train = len(mouse_idx_train)
N_test = len(mouse_idx_test)

############################
############################
##                        ##
##  Generate the weights  ##
##                        ##
############################
############################
weights_g = np.zeros(N_train)
weights_s = np.zeros(N_train)

# Positive and negative labels
indx_pos = ((aggression_idx_train==behaviorsnon1s.index(1))&(condition_idx_train==conditions.index(4)))
indx_neg1 = ((aggression_idx_train==behaviorsnon1s.index(2))&(condition_idx_train==conditions.index(4)))
indx_neg2 = ((aggression_idx_train==behaviorsnon1s.index(2))&(condition_idx_train==conditions.index(6)))
indx_neg3 = ((aggression_idx_train==behaviorsnon1s.index(2))&(condition_idx_train==conditions.index(8)))
indx_neg = indx_neg1|indx_neg2|indx_neg3

'''
##########################################################
# Generative weights balanced across mouse and condition #
##########################################################
mice_train = np.unique(mouse_idx_train)
nWindowsMouseCond = np.zeros((len(mice_train),2))
for i in range(len(mice_train)):
	idx_pos_mouse = (mouse_idx_train==mice_train[i])&indx_pos
	idx_neg_mouse = (mouse_idx_train==mice_train[i])&indx_neg
	nWindowsMouseCond[i,0] = np.sum(idx_pos_mouse)
	nWindowsMouseCond[i,1] = np.sum(idx_neg_mouse)

print(nWindowsMouseCond)
nWindowsMouseCond = 30 + nWindowsMouseCond

#Find the minimum number of windows
myMin = np.amin(nWindowsMouseCond)
for i in range(len(mice_train)):
	idx_pos_mouse = (mouse_idx_train==mice_train[i])&indx_pos
	idx_neg_mouse = (mouse_idx_train==mice_train[i])&indx_neg
	weights_g[idx_pos_mouse] = myMin/nWindowsMouseCond[i,0]
	weights_g[idx_neg_mouse] = myMin/nWindowsMouseCond[i,1]

#This just standardizes the weights so they aren't small
weights_g = weights_g/np.mean(weights_g)

'''
# Generate the generative weights, equal weighting for each mouse
mice_train = np.unique(mouse_idx_train)
nWindowsMouse = np.zeros(len(mice_train))
for i in range(len(mice_train)):
    nWindowsMouse[i] = np.sum(mouse_idx_train==mice_train[i]) + 40
#Find the minimum number of windows
myMin = np.amin(nWindowsMouse)
for i in range(len(mice_train)):
    weights_g[mice_train[i]==mouse_idx_train] = myMin/nWindowsMouse[i]

weights_g = weights_g/np.mean(weights_g)
#Supervision weights

y_train = np.zeros(N_train)
y_train[indx_pos] = 1
y_test = np.zeros(N_test)
indx_pos_test = ((aggression_idx_test==behaviorsnon1s.index(1))&(condition_idx_test==conditions.index(4)))
y_test[indx_pos_test] = 1


weights_s[indx_pos] = 1
weights_s[indx_neg1] = 1
weights_s[indx_neg2] = 1
weights_s[indx_neg3] = 1

#At the end make sure the mice in group 3 hae 0 weight
weights_s[mouse_idx_train==mice.index('Mouse049')] = 0
weights_s[mouse_idx_train==mice.index('Mouse057')] = 0
weights_s[mouse_idx_train==mice.index('Mouse131')] = 0
weights_s[mouse_idx_train==mice.index('Mouse3026')] = 0
weights_s[mouse_idx_train==mice.index('Mouse7975')] = 0
weights_s[mouse_idx_train==mice.index('Mouse7996')] = 0

wp = np.sum(weights_s[indx_pos])
wn1 = np.sum(weights_s[indx_neg1])
wn2 = np.sum(weights_s[indx_neg2])
wn3 = np.sum(weights_s[indx_neg3])
wn = wn1 + wn2 + wn3

wt = wp + wn1 + wn2 + wn3


weights_s[indx_pos] *= wt/wp
weights_s[indx_neg1] *= wt/wn1*.33
weights_s[indx_neg2] *= wt/wn2*.33
weights_s[indx_neg3] *= wt/wn3*.33


print(np.sum(weights_s[indx_pos]))
print(np.sum(weights_s[indx_neg1]))
print(np.sum(weights_s[indx_neg2]))
print(np.sum(weights_s[indx_neg3]))

print('>>>>>>>>>')
print(np.sum((y_train==1)*weights_s))
print(np.sum((y_train==0)*weights_s))
print('>>>>>>>>>')

# Make sure the weights are normalized

weights_s[weights_s>5] = 5

#weights_s = weights_s/np.mean(weights_s)
weights_s2 = np.zeros(N_train)
#weights_s[indx_pos] = 1
#weights_s[indx_neg] = 1
weights_s2[weights_s>0] = 1


print(np.min(weights_s),np.mean(weights_s),np.max(weights_s))
print(np.min(weights_g),np.mean(weights_g),np.max(weights_g))
nFact = 8
nIter = 20000

Ex = np.mean(X_train,axis=0)
reconRand_tr = np.mean((X_train-Ex)**2)
reconRand_te = np.mean((X_test-Ex)**2)
print('Random Reconstruction',np.mean((X_train-Ex)**2))
print('Random Reconstruction',np.mean((X_test-Ex)**2))

#mod_nmf = dp.NMF(nFact)
#S_train_nmf = mod_nmf.fit_transform(X_train)
#S_test_nmf = mod_nmf.transform(X_test)
#Xr_tr = np.dot(S_train_nmf,mod_nmf.components_)
#Xr_te = np.dot(S_test_nmf,mod_nmf.components_)
#print('Recon NMF',np.mean((Xr_tr-X_train)**2))
#print('Recon NMF',np.mean((Xr_te-X_test)**2))
#recon_nmf_tr = np.mean((Xr_tr-X_train)**2)
#recon_nmf_te = np.mean((Xr_te-X_test)**2)

mice_test = np.unique(mouse_idx_test)

model = NMF_logistic(nFact,nIter=nIter,LR=1e-3,mu=1.50,batchSize=100)
S_train = model.fit_transform(X_train,y_train,
                weights_s=weights_s,weights_g=weights_g)
S_test = model.transform(X_test)
components_ = model.components_
X_recon_tr = np.dot(S_train,components_)
X_recon_te = np.dot(S_test,components_)
print('Recon sNMF',np.mean((X_train-X_recon_tr)**2))
print('Recon sNMF',np.mean((X_test-X_recon_te)**2))
recon_snmf_tr = np.mean((X_recon_tr-X_train)**2)
recon_snmf_te = np.mean((X_recon_te-X_test)**2)

phi = model.Phi
print(phi)

print('First components')
print('Training ROC',roc_auc_score(y_train,S_train[:,0]))
print('Training ROC',roc_auc_score(y_test,S_test[:,0]))
roc_train = roc_auc_score(y_train,S_train[:,0])
roc_test = roc_auc_score(y_test,S_test[:,0])

idx1 = mice_test[0]==mouse_idx_test
idx2 = mice_test[1]==mouse_idx_test
idx3 = mice_test[2]==mouse_idx_test
print('ROC1',roc_auc_score(y_test[idx1],S_test[idx1,0]))
print('ROC2',roc_auc_score(y_test[idx2],S_test[idx2,0]))
print('ROC3',roc_auc_score(y_test[idx3],S_test[idx3,0]))

## Compare 1 vs 0

auc_test_1_0 = np.zeros(3)

indx_pos_test = ((aggression_idx_test==behaviorsnon1s.index(1))&(condition_idx_test==conditions.index(4)))
indx_neg1_test = ((aggression_idx_test==behaviorsnon1s.index(0))&(condition_idx_test==conditions.index(4)))
indx_neg2_test = ((aggression_idx_test==behaviorsnon1s.index(0))&(condition_idx_test==conditions.index(6)))
indx_neg3_test = ((aggression_idx_test==behaviorsnon1s.index(0))&(condition_idx_test==conditions.index(8)))
indx_neg_test = indx_neg1_test|indx_neg2_test|indx_neg3_test
indx_tot = indx_neg_test|indx_pos_test

y_test = np.zeros(len(aggression_idx_test))
y_test[indx_pos_test] = 1

y_test_sub = y_test[indx_tot]
mouse_idx_test_sub = mouse_idx_test[indx_tot]
S_test_sub = S_test[indx_tot]

idx1 = mice_test[0]==mouse_idx_test_sub
idx2 = mice_test[1]==mouse_idx_test_sub
idx3 = mice_test[2]==mouse_idx_test_sub
auc_test_1_0[0] = roc_auc_score(y_test_sub[idx1],S_test_sub[idx1,0])
auc_test_1_0[1] = roc_auc_score(y_test_sub[idx2],S_test_sub[idx2,0])
auc_test_1_0[2] = roc_auc_score(y_test_sub[idx3],S_test_sub[idx3,0])

print('1 vs 0')
print(auc_test_1_0)

## Compare 1 vs 2

auc_test_1_2 = np.zeros(3)

indx_pos_test = ((aggression_idx_test==behaviorsnon1s.index(1))&(condition_idx_test==conditions.index(4)))
indx_neg1_test = ((aggression_idx_test==behaviorsnon1s.index(2))&(condition_idx_test==conditions.index(4)))
indx_neg2_test = ((aggression_idx_test==behaviorsnon1s.index(2))&(condition_idx_test==conditions.index(6)))
indx_neg3_test = ((aggression_idx_test==behaviorsnon1s.index(2))&(condition_idx_test==conditions.index(8)))
indx_neg_test = indx_neg1_test|indx_neg2_test|indx_neg3_test
indx_tot = indx_neg_test|indx_pos_test

y_test = np.zeros(len(aggression_idx_test))
y_test[indx_pos_test] = 1

y_test_sub = y_test[indx_tot]
mouse_idx_test_sub = mouse_idx_test[indx_tot]
S_test_sub = S_test[indx_tot]

idx1 = mice_test[0]==mouse_idx_test_sub
idx2 = mice_test[1]==mouse_idx_test_sub
idx3 = mice_test[2]==mouse_idx_test_sub
auc_test_1_2[0] = roc_auc_score(y_test_sub[idx1],S_test_sub[idx1,0])
auc_test_1_2[1] = roc_auc_score(y_test_sub[idx2],S_test_sub[idx2,0])
auc_test_1_2[2] = roc_auc_score(y_test_sub[idx3],S_test_sub[idx3,0])

print('1 vs 2')
print(auc_test_1_2)

myDict = {}
myDict['recon_rand_tr'] = reconRand_tr
myDict['recon_rand_te'] = reconRand_te
#myDict['recon_nmf_tr'] = recon_nmf_tr
#myDict['recon_nmf_te'] = recon_nmf_te
myDict['recon_snmf_tr'] = recon_snmf_tr
myDict['recon_snmf_te'] = recon_snmf_te
myDict['losses_gen'] = model.losses_gen
myDict['losses_rec'] = model.losses_recon
myDict['losses_sup'] = model.losses_sup

myDict['S_train'] = S_train
myDict['S_test'] = S_test

myDict['Y_train'] = y_train
myDict['Y_test'] = y_test
myDict['roc_train'] = roc_train
myDict['roc_test'] = roc_test

myDict['phi'] = phi
myDict['components_'] = components_
myDict['A'] = model.A_enc
myDict['B'] = model.B_enc

myDict['auc_test_1_0'] = auc_test_1_0
myDict['auc_test_1_2'] = auc_test_1_2

pname = 'Trial1_best.p'
pickle.dump(myDict,open(pname,'wb'))


























