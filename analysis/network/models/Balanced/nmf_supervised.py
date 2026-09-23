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
from sklearn.model_selection import train_test_split
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

def predLoss(Y_true,Y_pred):
	yb = tf.squeeze(Y_true)
	yp = tf.squeeze(Y_pred)
	ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=yb,logits=yp)
	loss = tf.reduce_mean(ce)
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

	def _batchOS(self,X_pos,X_neg):
		bs = int(np.floor(self.batchSize/2))
		idx_pos_b = rand.choice(X_pos.shape[0],size=bs,replace=False)
		idx_neg_b = rand.choice(X_neg.shape[0],size=bs,replace=False)
		X_pos_b = X_pos[idx_pos_b]
		X_neg_b = X_neg[idx_neg_b]
		X_batch = np.vstack((X_pos_b,X_neg_b))
		return X_batch.astype(np.float32)


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
	
	def fit_transform(self,X_tot,X_sup,Y):
		#######################
		# Change X to float32 #
		#######################
		X_tot = X_tot.astype(np.float32)
		X_sup = X_sup.astype(np.float32)
		Y = Y.astype(np.float32)
		N,p = X_tot.shape

		##########################
		# Feature initialization #
		##########################

		#Get an NMF model
		mod_nmf = dp.NMF(self.n_components)
		SS = mod_nmf.fit_transform(X_sup)
		comp = mod_nmf.components_.astype(np.float32)

		W_init = softplus_inverse(comp)
		print('>>>>>>>>>')
		Xr = np.dot(SS,mod_nmf.components_)
		print(np.mean((Xr-X_sup)**2))
		print('>>>>>>>>>')
	
		#Initialize features
		W_r = tf.Variable(W_init.astype(np.float32))
		W_un = tf.nn.softplus(W_r)
		W = tf.math.l2_normalize(W_un,axis=1)
		WW = W.numpy()
		mod_nmf.components_ = WW.astype(np.float64)
		S_init = mod_nmf.transform(X_sup.astype(np.float64))

		print('>>>>>>a>>>')
		Xr = np.dot(S_init,mod_nmf.components_)
		print(np.mean((Xr-X_sup)**2))
		print('>>>>>>a>>>')

		#Shuffle the order from most to least predictive
		mod_lr = lm.LogisticRegression(C=.1)
		mod_lr.fit(S_init,Y)
		phi_abs = np.abs(mod_lr.coef_)

		print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
		print(mod_lr.coef_)
		print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

		order = np.argsort(phi_abs)[0]
		order = order[::-1]

		comp2 = mod_nmf.components_
		W_init = comp2[order]
		S_init = S_init[:,order]

		######################
		# Our linear encoder #
		######################
		Si = softplus_inverse(S_init.astype(np.float32))
		mod_lr = lm.LinearRegression()
		mod_lr.fit(X_sup,Si)
		B_enci = mod_lr.intercept_.astype(np.float32)
		A_enci = np.transpose(mod_lr.coef_.astype(np.float32))
		Spred = mod_lr.predict(X_sup)

		A_enc = tf.Variable(A_enci)
		B_enc = tf.Variable(B_enci)

		#####################
		# Get the optimizer #
		#####################
		trainable_variables_init = [A_enc,B_enc]

		optimizer_init = getOptimizer(self.LR,self.trainingMethod)
		optimizer_g = getOptimizer(self.LR,self.trainingMethod)
		optimizer_s = getOptimizer(self.LR,self.trainingMethod)
		optimizer_p = getOptimizer(10*self.LR,self.trainingMethod)

		###################
		# Losses we track #
		###################
		losses_gen = np.zeros(self.nIter)
		losses_recon = np.zeros(self.nIter)
		losses_sup = np.zeros(self.nIter)

		##########################################################
		# Initialize encoder and features to good starting value #
		##########################################################
		N_init = 1000000
		losses_init = np.zeros(N_init)
		cont = True
		jj = 0
		os_likelihood = 100000

		Xi_train,Xi_test, = train_test_split(X_sup,test_size=.2,
												random_state=42)
		Xi_train = Xi_train.astype(np.float32)
		Xi_test = Xi_test.astype(np.float32)
		while cont:
			X_batch = self._batchX(Xi_train)
			with tf.GradientTape() as tape:
				W_un = tf.nn.softplus(W_r)
				W = tf.math.l2_normalize(W_un,axis=1)
				S_latent = tf.matmul(X_batch,A_enc) + B_enc
				S_ = tf.nn.softplus(S_latent)
				X_recon = tf.matmul(S_,W)
				loss = L2_loss(X_batch,X_recon)
			grad = tape.gradient(loss,trainable_variables_init)
			optimizer_init.apply_gradients(zip(grad,
												trainable_variables_init))
			losses_init[jj] = loss.numpy()
			jj += 1
			if jj % 50 == 0:
				S_latent = tf.matmul(Xi_test,A_enc) + B_enc
				S_ = tf.nn.softplus(S_latent)
				X_recon = tf.matmul(S_,W)
				print('>>>>>>>>>>>>')
				print(loss.numpy())
				loss = L2_loss(Xi_test,X_recon)
				print(jj,loss.numpy())
				if ((loss.numpy() < os_likelihood)|(jj<1500)):
					os_likelihood = loss.numpy()
				else:
					cont = False
					self.stop = jj

		self.losses_init = losses_init[:self.stop]


		#########################################################
		S_tr = tf.nn.softplus(tf.matmul(X_sup,A_enc) + B_enc)
		S_tr2 = S_tr.numpy()
		model_lm = lm.LogisticRegression()
		print(".>>>>>>>>>>>>>>>>")
		model_lm.fit(S_tr2,Y)
		print(model_lm.coef_)
		print(".>>>>>>>>>>>>>>>>")


		# Initialize supervision parameters
		phis = np.zeros(self.n_components)
		for i in range(self.n_components):
			model_lm = lm.LogisticRegression()
			model_lm.fit(np.atleast_2d(S_tr2[:,i]).T,Y)
			print(".>>>>>>>>>>>>>>>>")
			print(model_lm.coef_)
			print(".>>>>>>>>>>>>>>>>")
			phis[i] = model_lm.coef_

		order = np.argsort(np.abs(phis))
		print(order)
		order = order[::-1]

		A_new = A_enc.numpy()
		A_new = A_new[:,order]
		B_new = B_enc.numpy()
		print(A_new.shape)
		print(B_new.shape)
		B_new = B_new[order]

		A_enc = tf.Variable(A_new.astype(np.float32))
		B_enc = tf.Variable(B_new.astype(np.float32))

		S_tr = tf.nn.softplus(tf.matmul(X_sup,A_enc) + B_enc)
		S_tr2 = S_tr.numpy()
		model_lm = lm.LogisticRegression()
		print(".>>>>>>>>>>>>>>>>")
		model_lm.fit(S_tr2[:,:self.n_blessed],Y)
		print(model_lm.coef_)
		print(".>>>>>>>>>>>>>>>>")

		B_init = model_lm.intercept_.astype(np.float32)
		phi_init = model_lm.coef_.astype(np.float32)

		Phi_ = tf.Variable(np.atleast_2d(phi_init).T)
		B_ = tf.Variable(B_init) 
		trainable_variables_g = [W_r,A_enc,B_enc]
		trainable_variables_s = [A_enc,B_enc,Phi_,B_]
		trainable_variables_p = [Phi_,B_]

		bbs = int(np.floor(self.batchSize/2))
		Y_batch_b = np.ones(2*bbs)
		Y_batch_b[bbs:] = 0
		Y_batch_b = Y_batch_b.astype(np.float32)
		idx_pos = Y==1
		idx_neg = Y==1
		X_pos = X_sup[idx_pos]
		X_neg = X_sup[idx_neg]
		print('Hello')
		print(phi_init)
		print(B_init)

		############################
		# Actually train the model #
		############################
		for t in trange(self.nIter):

			for ii in range(1):
				X_batch = self._batchX(X_tot)

				#This is the generative part, regular sampling
				with tf.GradientTape() as tape:
					W_un = tf.nn.softplus(W_r)
					W = tf.math.l2_normalize(W_un,axis=1)
					S_latent = tf.matmul(X_batch,A_enc) + B_enc
					S_ = tf.nn.softplus(S_latent)

					#Reconstruction loss
					X_recon = tf.matmul(S_,W)
					loss_recon = L2_loss(X_batch,X_recon)

					#Total loss
					loss = loss_recon 

				# Estimate the gradients and apply them with optimizer
				grad_g = tape.gradient(loss,trainable_variables_g)
				optimizer_g.apply_gradients(zip(grad_g,
											trainable_variables_g))

			# Save the losses 

			#This is the oversampled part
			for jj in range(10):
				X_batch = self._batchOS(X_pos,X_neg)
				with tf.GradientTape() as tape:
					W_un = tf.nn.softplus(W_r)
					W = tf.math.l2_normalize(W_un,axis=1)
					S_latent = tf.matmul(X_batch,A_enc) + B_enc
					S_ = tf.nn.softplus(S_latent)

					#Reconstruction loss
					X_recon = tf.matmul(S_,W)
					loss_recon = L2_loss(X_batch,X_recon)
				
					#Supervision loss
					Y_pred = tf.matmul(S_[:,:self.n_blessed],Phi_) + B_
					loss_sup = predLoss(Y_batch_b,Y_pred)

					loss = loss_recon + self.mu*loss_sup

				# Estimate the gradients and apply them with optimizer
				grad_s = tape.gradient(loss,trainable_variables_s)
				optimizer_s.apply_gradients(zip(grad_s,
												trainable_variables_s))



			losses_gen[t] = loss.numpy()
			losses_recon[t] = loss_recon.numpy()
			losses_sup[t] = loss_sup.numpy()

			#This is the oversampled part
			for kk in range(10):
				X_batch = self._batchOS(X_pos,X_neg)
				with tf.GradientTape() as tape:
					S_latent = tf.matmul(X_batch,A_enc) + B_enc
					S_ = tf.nn.softplus(S_latent)

					#Supervision loss
					Y_pred = tf.matmul(S_[:,:self.n_blessed],Phi_) + B_
					loss_sup = predLoss(Y_batch_b,Y_pred)

				# Estimate the gradients and apply them with optimizer
				grad_p = tape.gradient(loss_sup,trainable_variables_p)
				optimizer_p.apply_gradients(zip(grad_p,
												trainable_variables_p))

		self.losses_gen = losses_gen
		self.losses_sup = losses_sup
		self.losses_recon = losses_recon

		#Project this data
		S_r = tf.nn.softplus(tf.matmul(X_sup,A_enc) + B_enc)
		Scores = S_r.numpy()

		# Save the variables of interest 
		self.components_ = W.numpy()

		self.A_enc = A_enc.numpy()
		self.B_enc = B_enc.numpy()

		self.Phi = Phi_.numpy()
		self.B_ = B_.numpy()

		self.reconstruction_err_ = _beta_divergence(X_sup,Scores,
										self.components_,beta=self.beta)

		return Scores
	
	def transform(self,X):
		S_latent = np.dot(X,self.A_enc) + self.B_enc
		S_l = tf.nn.softplus(S_latent)
		return S_l.numpy()
	
