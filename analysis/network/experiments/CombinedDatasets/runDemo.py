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


######################################################################
## This section keeps the code from soaking up your entire computer ##
######################################################################

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2024)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

sys.path.append('/home/austin/DataAnalysis')
from data_tools import load_data


# This line actually loads in your data
fnm='/media/austin/ThickBoy__1/DataAgression_Granger2/Aggression_sub_12.mat'
power,coherence,granger,labels = load_data(fnm,fBounds=(1,56),
                        feature_list=['power','coherence','granger'])

#Get the labels we care about
myLabel = labels['windows']
mouse = np.asarray(myLabel['mouse'])
group = np.asarray(myLabel['group'])
expDate = np.asarray(myLabel['expDate'])
behavior = np.asarray(myLabel['behavior'])
behaviornon1 = np.asarray(myLabel['behaviornon1'])
time = np.asarray(myLabel['time'])
condition = np.asarray(myLabel['condition'])
N = len(mouse)


#######################################################################
## This is one big section where I get the labels for the aggression ##
## project and divide into training and test sets                    ##
#######################################################################

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

#Stack all the covariates together
X = np.hstack((power,coherence,granger))
X = X[indx_tot]

X_train = X[training_set_idx==1]
m_train = mouse[training_set_idx==1]
y_train = y[training_set_idx==1]

X_test = X[training_set_idx==0]
m_test = mouse[training_set_idx==0]
y_test = y[training_set_idx==0]


## Set the supervision strength and number of factors
mu = float(sys.argv[1])
nFact = 8

nIter = 20000
model = NMF_logistic(nFact,nIter=nIter,LR=1e-3,mu=mu,batchSize=100)
S_train = model.fit_transform(X_train_tot,y_train_tot)
S_train = model.transform(X_train)
S_test = model.transform(X_test)
S_train_new = model.transform(X_train_new)
S_test_new = model.transform(X_test_new)
components_ = model.components_


## Collect the results we care about

myDict = {}
myDict['S_train'] = S_train
myDict['S_test'] = S_test
myDict['S_train_new'] = S_train_new
myDict['S_test_new'] = S_test_new
myDict['components'] = components_
myDict['phi'] = model.Phi
myDict['A_enc'] = model.A_enc
myDict['B_enc'] = model.B_enc

## Savename 
pname = 'Unbalanced_Elastic_12_enc_' + str(mu) + '.p'
pickle.dump(myDict,open(pname,'wb'))




























