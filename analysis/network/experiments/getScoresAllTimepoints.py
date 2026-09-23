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


sys.path.append('/home/austin/DataAnalysis')
from data_tools import load_data

fnm='/media/austin/ThickBoy__1/DataAgression_Granger2/Aggression_sub_12.mat'
power,coherence,granger,labels = load_data(fnm,fBounds=(1,56),
                        feature_list=['power','coherence','granger'])


myLabel = labels['windows']
mouse = np.asarray(myLabel['mouse'])
group = np.asarray(myLabel['group'])
expDate = np.asarray(myLabel['expDate'])
behavior = np.asarray(myLabel['behavior'])
behaviornon1 = np.asarray(myLabel['behaviornon1'])
time = np.asarray(myLabel['time'])
condition = np.asarray(myLabel['condition'])


granger = np.exp(granger)
granger[granger>10] = 10
power = power*10
power[power>6] = 6
X = np.hstack((power,coherence,granger))

myDict = pickle.load(open('Unbalanced_Elastic_12_enc_1.0.p','rb'))
A_enc = myDict['A_enc']
B_enc = myDict['B_enc']

scores_l = np.dot(X,A_enc) + B_enc
scores_tf = tf.nn.softplus(scores_l)
scores = scores_tf.numpy()

saveDict = {}
saveDict['mouse'] = mouse
saveDict['group'] = group
saveDict['expDate'] = expDate
saveDict['time'] = time
saveDict['condition'] = condition
saveDict['behavior'] = behavior
saveDict['behaviornon1'] = behaviornon1
saveDict['scores'] = scores

pickle.dump(saveDict,open('Aggression_sub_12_scores.p','wb'))

