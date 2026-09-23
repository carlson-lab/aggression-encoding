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
from nmf_joint import NMF_logistic

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

model_nmf = dp.NMF(nFact)
S_train = model_nmf.fit_transform(X_train)
S_test = model_nmf.transform(X_test)

Xr_tr = np.dot(S_train,model_nmf.components_)
Xr_te = np.dot(S_test,model_nmf.components_)

print(np.mean((Xr_tr-X_train)**2))
print(np.mean((Xr_te-X_test)**2))

myDict_r = {'model_nmf':model_nmf}
pickle.dump(myDict_r,open('SKLEARN_NMF.p','wb'))




