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

def L2_loss(X_true,X_est):
	diff = X_true - X_est
	diff2 = tf.square(diff)
	loss = tf.reduce_mean(diff2)
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
	
	def _batch(self,X,Y):
		N = X.shape[0]
		idx = rand.choice(N,size=self.batchSize,replace=False)
		X_batch = X[idx]
		Y_batch = Y[idx]
		return X_batch,Y_batch

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
	
	def fit_transform(self,X,Y):
		#######################
		# Change X to float32 #
		#######################
		X = X.astype(np.float32)
		Y = Y.astype(np.float32)
		N,p = X.shape

		##########################
		# Feature initialization #
		##########################
		mod_nmf = dp.NMF(self.n_components)
		SS = mod_nmf.fit_transform(X)
		comp = mod_nmf.components_.astype(np.float32)
		W_init = softplus_inverse(comp)
		W_ri = tf.Variable(W_init.astype(np.float32))
		W_uni = tf.nn.softplus(W_ri)
		Wi = tf.math.l2_normalize(W_uni,axis=1)

		#Initial encoder
		WW = Wi.numpy()
		mod_nmf.components_ = WW.astype(np.float64)
		S_init = mod_nmf.transform(X.astype(np.float64))
		Si = softplus_inverse(S_init.astype(np.float32))
		mod_lr = lm.LinearRegression()
		mod_lr.fit(X,Si)
		B_enci = mod_lr.intercept_.astype(np.float32)

		A_enci = np.transpose(mod_lr.coef_.astype(np.float32))
		A_enc_i = tf.Variable(A_enci)
		B_enc_i = tf.Variable(B_enci)

		#####################
		# Get the optimizer #
		#####################
		trainable_variables_init = [A_enc_i,B_enc_i,W_ri]
		optimizer_init = getOptimizer(self.LR,self.trainingMethod)
		N_init = 5#000
		losses_init = np.zeros(N_init)

		##########################################################
		# Initialize encoder and features to good starting value #
		##########################################################
		for t in trange(N_init):
			X_batch,_ = self._batch(X,Y)
			with tf.GradientTape() as tape:
				W_uni = tf.nn.softplus(W_ri)
				Wi = tf.math.l2_normalize(W_uni,axis=1)
				S_latent = tf.matmul(X_batch,A_enc_i) + B_enc_i
				S_ = tf.nn.softplus(S_latent)
				X_recon = tf.matmul(S_,Wi)
				loss = L2_loss(X_batch,X_recon)
			grad = tape.gradient(loss,trainable_variables_init)
			optimizer_init.apply_gradients(zip(grad,trainable_variables_init))
			losses_init[t] = loss.numpy()
		self.losses_init1 = losses_init

		#########################################
		# Reorder according to the coefficients #
		#########################################

		# Initialize supervision parameters
		S_tr = tf.nn.softplus(tf.matmul(X,A_enc_i) + B_enc_i)
		S_tr2 = S_tr.numpy()
		model_lm = lm.LogisticRegression()
		model_lm.fit(S_tr2,Y)
		B_init = model_lm.intercept_.astype(np.float32)
		phis = model_lm.coef_

		ids = np.argsort(phis)[0]#ids[0] most negative ids[-1] most positive
		Wr_old = W_ri.numpy()
		Wr_new = np.zeros(WW.shape)
		print(phis)
		Wr_new[0] = Wr_old[ids[0]]# Most negative
		Wr_new[1] = Wr_old[ids[-1]]# Most positive 
		Wr_new[2:] = Wr_old[ids[1:-1]] #The remainder

		A_enci = np.zeros(A_enc_i.numpy().shape)
		B_enci = np.zeros(B_enc_i.numpy().shape)
		Anci = A_enc_i.numpy()
		Bnci = B_enc_i.numpy()
		A_enci[:,0] = Anci[:,ids[0]]
		A_enci[:,1] = Anci[:,ids[-1]]
		A_enci[:,2:] = Anci[:,ids[1:-1]]

		B_enci[0] = Bnci[ids[0]]
		B_enci[1] = Bnci[ids[-1]]
		B_enci[2:] = Bnci[ids[1:-1]]

		A_enc = tf.Variable(A_enci.astype(np.float32))
		B_enc = tf.Variable(B_enci.astype(np.float32))

		W_r = tf.Variable(Wr_new.astype(np.float32))
		W_un = tf.nn.softplus(W_ri)
		W = tf.math.l2_normalize(W_uni,axis=1)
		'''
		WW = W.numpy()
		mod_nmf.components_ = WW.astype(np.float64)
		S_init = mod_nmf.transform(X.astype(np.float64))
		Si = softplus_inverse(S_init.astype(np.float32))
		mod_lr = lm.LinearRegression()
		mod_lr.fit(X,Si)
		B_enci = mod_lr.intercept_.astype(np.float32)
		A_enci = np.transpose(mod_lr.coef_.astype(np.float32))
		A_enc = tf.Variable(A_enci.astype(np.float32))
		B_enc = tf.Variable(B_enci.astype(np.float32))


		trainable_variables_init2 = [A_enc,B_enc]
		optimizer_init2 = getOptimizer(self.LR,self.trainingMethod)
		losses_init2 = np.zeros(N_init)

		##########################################################
		# Initialize encoder and features to good starting value #
		##########################################################
		for t in trange(N_init):
			X_batch,_ = self._batch(X,Y)
			with tf.GradientTape() as tape:
				W_un = tf.nn.softplus(W_r)
				W = tf.math.l2_normalize(W_un,axis=1)
				S_latent = tf.matmul(X_batch,A_enc) + B_enc
				S_ = tf.nn.softplus(S_latent)
				X_recon = tf.matmul(S_,W)
				loss = L2_loss(X_batch,X_recon)
			grad = tape.gradient(loss,trainable_variables_init2)
			optimizer_init2.apply_gradients(zip(grad,
										trainable_variables_init2))
			losses_init2[t] = loss.numpy()
		self.losses_init2 = losses_init2
		'''

		#Now once again we get the logistic coefficients
		S_tr = tf.nn.softplus(tf.matmul(X,A_enc) + B_enc)
		S_tr2 = S_tr.numpy()
		model_lm = lm.LogisticRegression()
		model_lm.fit(S_tr2[:,:2],Y)
		phi_init = model_lm.coef_.astype(np.float32)
		print(phi_init)

		#Define the variables 
		Phi_l = tf.Variable(np.atleast_2d(phi_init).T)
		B_init = model_lm.intercept_.astype(np.float32)
		B_ = tf.Variable(B_init) 

		trainable_variables_g = [W_r,A_enc,B_enc,Phi_l,B_]
		optimizer_g = getOptimizer(self.LR,self.trainingMethod)

		###################
		# Losses we track #
		###################
		losses_gen = np.zeros(self.nIter)
		losses_recon = np.zeros(self.nIter)
		losses_sup = np.zeros(self.nIter)

		ss = np.ones((2,1))
		ss[0,0] = -1

		############################
		# Actually train the model #
		############################
		for t in trange(self.nIter):
			X_batch,Y_batch = self._batch(X,Y)
			with tf.GradientTape() as tape:
				W_un = tf.nn.softplus(W_r)
				W = tf.math.l2_normalize(W_un,axis=1)
				S_latent = tf.matmul(X_batch,A_enc) + B_enc
				S_ = tf.nn.softplus(S_latent)

				#Reconstruction loss
				X_recon = tf.matmul(S_,W)
				loss_recon = L2_loss(X_batch,X_recon)

				Phi_a = tf.math.abs(Phi_l) + 1e-2
				Phi_ = tf.multiply(ss,Phi_a)

				#Supervision loss
				Y_pred = tf.matmul(S_[:,:2],Phi_) + B_
				ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.squeeze(Y_batch),
										logits=tf.squeeze(Y_pred))
				loss_sup = tf.reduce_mean(ce)

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
		S_r = tf.nn.softplus(tf.matmul(X,A_enc) + B_enc)
		Scores = S_r.numpy()

		# Save the variables of interest 
		self.components_ = W.numpy()

		self.A_enc = A_enc.numpy()
		self.B_enc = B_enc.numpy()

		self.Phi = Phi_.numpy()
		self.B_ = B_.numpy()

		return Scores
	
	def transform(self,X):
		S_latent = np.dot(X,self.A_enc) + self.B_enc
		S_l = tf.nn.softplus(S_latent)
		return S_l.numpy()
	
