# https://github.com/stiandb/Thesis/blob/c3f3243fdeed6a099a16006d2bf529e2ceedc2b4/Deep-Learning/QDNN.py
import sys
sys.path.append('../')
from dl_utils import *
from scipy.optimize import minimize, Bounds
from utils import *
import numpy as np
import qiskit as qk


class QDNN(Utils):
	def __init__(self,layers=None,loss_fn = None,classification=False):
		"""
		Inputs:
			layers (list) - List containing layers for dense neural network
			loss_fn (callable) - loss_fn(y_pred, y) returns the loss where y_pred is 
									the prediction and y is the actual target variable
			classification (boolean) - If True, we are dealing with a classification model
		"""
		self.classification=classification
		self.layers = layers
		self.loss_fn = loss_fn
		self.loss_train = []
		self.loss_val = []
		self.w_opt = None
		self.first_run = True

	def forward(self,X):
		"""
		Input:
			X (numpy array) - Numpy array of dimension [n,p], where n is the number of samples 
								and p is the number of predictors
		Output:
			X (numpy array) - The output from the neural network.
		"""

		for layer in self.layers:
			if type(layer) is list:
				X_ = []
				idx=0
				for sub_layer in layer:
					X_.append(sub_layer(X[:,idx:idx+sub_layer.n_qubits]))
					idx += sub_layer.n_qubits
				X = np.hstack(X_)
			else:
				X = layer(X)
		if self.classification:
			X = X/(np.sum(X,axis=1).reshape(X.shape[0],1) + 1e-14)
		return(X)

	def fit(self,X,y,X_val=None,y_val=None,method='Powell',max_iters = 10000,max_fev=None,tol=1e-14,seed=None,bounds=None,print_loss=False,w_init = None):
		"""
		Uses classical optimization to train the neural network.
		Input:
			X (numpy array) - design matrix for the problem
			y (numpy array) - target variable for the problem
			X_val (None or numpy array) - Validation design matrix
			y_val (None or numpy array) - Validation target variables
			method (str)    - the classical optimization method to use
			max_iters (int)- The maximum number of iterations for the classical
								optimization.
			max_fev (int) - The maximum number of function evaluations for the classical
								optimization
			tol (float)   - Tolerance for convergence condition
			seed (int or None) - The seed for optimizer
			bounds (Bounds object) - If method allows for bounds, set them with this argument
			print_loss (boolean) - If True, loss will be printed every function evaluation
			w_init (None or numpy 1d array) - The initial weights for optimizer
			
		"""
		options = {'disp':True,'maxiter':max_iters}
		if not seed is None:
			np.random.seed(seed)
		if not max_fev is None:
			options['maxfev'] = max_fev
		self.n_weights = 0
		for layer in self.layers:
			if type(layer) is list:
				for sub_layer in layer:
					self.n_weights += sub_layer.w_size
			else:
				self.n_weights += layer.w_size
		if w_init is None:
			w = 1+0.1*np.random.randn(self.n_weights)
		else:
			w = w_init
			self.w_opt = w_init.copy()
		if bounds is None:
			w = minimize(self.calculate_loss,w,args=(X,y,X_val,y_val,print_loss),method=method,options=options,tol=tol).x
		else:
			w = minimize(self.calculate_loss,w,args=(X,y,X_val,y_val,print_loss),method=method,options=options,tol=tol,bounds=Bounds(bounds[0],bounds[1])).x
		self.set_weights(self.w_opt)
		self.loss_train = np.array(self.loss_train)
		self.loss_val = np.array(self.loss_val)

	def calculate_loss(self,w,X,y,X_val=None,y_val=None,print_loss=False):
		"""
		Input:
			w (numpy array) - One dimensional array containing 
								all network weights
			X (numpy array) - design matrix for the problem
			y (numpy array) - target variable for the problem
		Output:
			cost (float) 	- The loss for the data.
		"""
		self.set_weights(w)
		y_pred = self.forward(X)
		cost_train = self.loss_fn(y_pred,y)
		if X_val is None:
			if not self.first_run and (cost_train < np.min(np.array(self.loss_train))):
				self.w_opt = w.copy()
				np.save('w_opt.npy',self.w_opt)
			self.loss_train.append(cost_train)
			if print_loss:
				try:
					print('Training loss: ',cost_train, 'Min loss: ', min(self.loss_train))
				except:
					print('Training loss: ',cost_train)
		else:
			y_val_pred = self.forward(X_val)
			cost_val = self.loss_fn(y_val_pred,y_val)
			self.loss_val.append(cost_val)
			self.loss_train.append(cost_train)
			if not self.first_run and (cost_val < np.min(np.array(self.loss_val))):
				self.w_opt = w.copy()
			if print_loss:
				print('Training loss: ',cost_train, ' Validation loss: ',cost_val,'Min val loss:',min(self.loss_val))
		self.first_run = False
		return(cost_train)



