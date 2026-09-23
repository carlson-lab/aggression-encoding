'''
Creator:
    Austin "The Man" Talbot
Creation Date:
    11/16/2019
Version history
---------------
Version 1.0
Objects
-------
sNMF_L1
sNMF_blessed
References
----------
https://www.tensorflow.org/

https://scikit-learn.org/stable/auto_examples/decomposition/plot_faces_decomposition.html#sphx-glr-auto-examples-decomposition-plot-faces-decomposition-py
'''
import numpy as np
import numpy.random as rand
import sys,os
import tensorflow as tf
from tensorflow import keras
from datetime import datetime as dt
import numpy.linalg as la
import pickle
import time
from sklearn.utils.extmath import randomized_svd,squared_norm
from sklearn import decomposition as dp
from sklearn import linear_model as lm
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import average_precision_score,roc_auc_score
from sklearn.metrics import roc_curve,auc
from tqdm import trange
from ml_base import norm,nndsvda_init,getOptimizer,activateVariable
from ml_base import softplus_inverse,_beta_divergence,np_softplus
from ml_base import np_softplus5

version = '1.0'

def L2_loss(X_true,X_est,weights):
	diff = X_true - X_est
	diff2 = tf.square(diff)
	loss_uw = tf.reduce_mean(diff2,axis=1)
	lw = tf.math.multiply(loss_uw,weights)
	loss = tf.reduce_mean(lw)
	return loss

def predLoss(Y_true,Y_pred,weights):
	yb = tf.squeeze(Y_true)
	yp = tf.squeeze(Y_pred)
	ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=yb,logits=yp)
	wce = tf.math.multiply(ce,weights)
	loss = tf.reduce_mean(wce)
	return loss


def softplus_inverse(X):
    out = np.zeros(X.shape)
    out[X>10] = X[X>10]
    out[X<.0001] = -9.21
    out[(X>.0001)&(X<10)] = np.log(np.exp(X[(X>.0001)&(X<10)])-1)
    return out

class NMF_base(object):

	def __init__(self,n_components,nIter=50000,LR=1e-5,name='NMF_g',
							dirName='./tmp',device=0,gpuMem=1024,
							decoderActiv='softplus',factorActiv='softplus',
							trainingMethod='Nadam',beta='frobenius',
							mu=1.0,batchSize=100):
		self.n_components = int(n_components)
		self.nIter = int(nIter)
		self.LR = float(LR)
		self.beta = beta
		self.batchSize = batchSize

		self.device = int(device)
		self.gpuMem = int(gpuMem)

		self.name = str(name)
		self.dirName = str(dirName)

		self.trainingMethod = str(trainingMethod)
		self.factorActiv = factorActiv
		self.decoderActiv = decoderActiv

		self.creationDate = dt.now()
		self.version = version

		self.mu = float(mu)
	
	def saveModel(self,saveName):
		myDict = {'model':self}
		pickle.dump(myDict,open(saveName,'wb'))
	
	def saveComponents(self,saveName,method='matlab'):
		if method == 'matlab':
			myDict = {'components':self.components_}
			scipy.io.savemat(saveName,myDict)
		elif method == 'csv':
			np.savetxt(saveName,self.components_,fmt='%0.8f',delimiter=',')
		else:
			print('Unrecognized save method %s'%method)
	
	def _batch(self,X,Y,wg,ws):
		N = X.shape[0]
		idx = rand.choice(N,size=self.batchSize,replace=False)
		X_batch = X[idx]
		Y_batch = Y[idx]
		wg_batch = wg[idx]
		ws_batch = ws[idx]
		return X_batch,Y_batch,wg_batch,ws_batch

	def _batchX(self,X):
		N = X.shape[0]
		idx = rand.choice(N,size=self.batchSize,replace=False)
		X_batch = X[idx]
		return X_batch

class NMF_logistic(NMF_base):
	
	def __init__(self,n_components,nIter=50000,LR=1e-3,name='jNMF_lm',
							dirName='./tmp',device=0,gpuMem=1024,
							decoderActiv='softplus',factorActiv='softplus',
							trainingMethod='Nadam',beta='frobenius',
							n_blessed=1,mu=1.0,batchSize=100):
		NMF_base.__init__(self,n_components,nIter=nIter,LR=LR,name=name,
							trainingMethod=trainingMethod,dirName=dirName,
							factorActiv=factorActiv,device=device,beta=beta,
							decoderActiv=decoderActiv,mu=mu,
							batchSize=batchSize)
		self.n_blessed = int(n_blessed)
	
	def fit_transform(self,X,Y,weights_s=None,weights_g=None):
		##############################
		# Change matrices to float32 #
		##############################
		X = X.astype(np.float32)
		Y = Y.astype(np.float32)
		if weights_s is not None:
			weights_s = np.squeeze(weights_s)
			weights_s = weights_s.astype(np.float32)
		else:
			weights_s = np.ones(X.shape[0]).astype(np.float32)
		if weights_g is not None:
			weights_g = np.squeeze(weights_g)
			weights_g = weights_g.astype(np.float32)
		else:
			weights_g = np.ones(X.shape[0]).astype(np.float32)
		N,p = X.shape

		##########################
		# Feature initialization #
		##########################
		mod_nmf = dp.NMF(self.n_components)
		SS = mod_nmf.fit_transform(X)
		comp = mod_nmf.components_.astype(np.float32)
		W_init = softplus_inverse(comp)

		W_r = tf.Variable(W_init.astype(np.float32))
		W_un = tf.nn.softplus(W_r)
		W = tf.math.l2_normalize(W_un,axis=1)

		encoder = keras.Sequential([
				keras.layers.InputLayer(input_shape=p),
				keras.layers.Dense(60,activation='elu'),
				keras.layers.Dense(self.n_components,activation='softplus')
		])

		trainable_variables_init = encoder.trainable_variables
		optimizer_init2 = getOptimizer(self.LR,self.trainingMethod)
		N_init = 1000
		losses_init = np.zeros(N_init)

		for t in trange(N_init):
			X_batch,_,wg_batch,_ = self._batch(X,Y,weights_g,weights_s)
			with tf.GradientTape() as tape:
				S_ = encoder(X_batch)
				X_recon = tf.matmul(S_,W)
				loss = L2_loss(X_batch,X_recon,wg_batch)
			grad = tape.gradient(loss,trainable_variables_init)
			optimizer_init2.apply_gradients(zip(grad,trainable_variables_init))
			losses_init[t] = loss.numpy()
		self.losses_init2 = losses_init

		#Define the variables 
		Phi_ = tf.Variable(rand.randn(1,1).astype(np.float32))
		B_ = tf.Variable(rand.randn(1,1).astype(np.float32))

		trainable_variables_g = [W_r,Phi_,B_] + encoder.trainable_variables
		optimizer_g = getOptimizer(self.LR,self.trainingMethod)

		###################
		# Losses we track #
		###################
		losses_gen = np.zeros(self.nIter)
		losses_recon = np.zeros(self.nIter)
		losses_sup = np.zeros(self.nIter)

		############################
		# Actually train the model #
		############################
		for t in trange(self.nIter):
			X_batch,Y_batch,wg_batch,ws_batch = self._batch(X,Y,
													weights_g,weights_s)
			with tf.GradientTape() as tape:
				W_un = tf.nn.softplus(W_r)
				W = tf.math.l2_normalize(W_un,axis=1)
				S_ = encoder(X_batch)

				#Reconstruction loss
				X_recon = tf.matmul(S_,W)
				loss_recon = L2_loss(X_batch,X_recon,wg_batch)

				#Supervision loss
				Y_pred = tf.matmul(S_[:,:1],Phi_) + B_
				loss_sup = predLoss(Y_batch,Y_pred,ws_batch)

				#Total loss
				loss = loss_recon + self.mu*loss_sup 

			# Estimate the gradients and apply them with optimizer
			grad_g = tape.gradient(loss,trainable_variables_g)
			optimizer_g.apply_gradients(zip(grad_g,
										trainable_variables_g))
			
			# Save the losses 
			losses_gen[t] = loss.numpy()
			losses_sup[t] = loss_sup.numpy()
			losses_recon[t] = loss_recon.numpy()

		self.losses_gen = losses_gen
		self.losses_sup = losses_sup
		self.losses_recon = losses_recon

		#Project this data
		S_r = encoder(X)
		Scores = S_r.numpy()

		# Save the variables of interest 
		self.components_ = W.numpy()

		self.Phi = Phi_.numpy()
		self.B_ = B_.numpy()

		return Scores,encoder
	
	def transform(self,X,encoder):
		S_l = encoder(X)
		return S_l.numpy()
	
