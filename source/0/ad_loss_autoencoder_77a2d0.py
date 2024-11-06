# https://github.com/erenaykrcn/HAE/blob/2fedfe5a096592549e75ceacb8bb1f13375ad2e0/modules/AD_loss_autoencoder/AD_loss_autoencoder.py
import os
dirname = os.path.dirname(__file__)

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

import numpy as np
from qiskit import Aer
from sklearn.ensemble import IsolationForest

import sys
sys.path.append(os.path.join(dirname, '../../'))
from modules.classical_autoencoder.classical_autoencoder import ClassicalAutoencoder
from modules.preprocessing.preprocessing import preprocess, sample_training_data, sample_test_data
from modules.AD_loss_autoencoder.hybrid import Hybrid


class ADLossAutoencoder():
	def __init__(self, qc_index, epochs=50, learningRate=1e-3, n_samples=50, loss_value_classical=0.023, shift=np.pi / 2, approach=0):
		"""
			Approach 1 and 0 use different fitting methods for the IsolationForest, whose decision_function method
			is used as the interpreter function of our Quantum Neural Network. 
			
			-> Approach 0 (Default): Latent space representation of the normal data is fitted to the IsolationForest and 
			same fitted model is used for all epochs. 

			-> Approach 1: Latent space representation of the normal data is encoded to the QC with the current theta params
			of the current epoch and the outcome of the QC is fitted into the IsolationForest. Hence, every epoch uses a 
			different interpreter function. Due to the processing of fitting data every epoch through the QC, it is slower.
		"""


		self.approach = approach

		path_cl = f'../../data/training_results/classical/training_result_loss_{loss_value_classical}.pt'
		self.cae = ClassicalAutoencoder()
		self.cae.load_state_dict(torch.load(os.path.join(dirname, path_cl)))
		self.cae.eval()

		if approach == 0:
			fit_data = Variable(torch.FloatTensor(sample_training_data(1000)[0]))
		else:
			fit_data = Variable(torch.FloatTensor(sample_training_data(250)[0]))
		fit_data_latent = self.cae.get_latent_space_state(fit_data).tolist()
		self.if_model = IsolationForest().fit(fit_data_latent)

		self.hybrid = Hybrid(backend=Aer.get_backend('aer_simulator'), qc_index=qc_index, shots=100, 
			shift=shift, if_model=None if approach else self.if_model,
			fit_data=fit_data_latent if approach else None
			)

		self.epochs = epochs
		self.learningRate = learningRate

		# TODO: Try different loss func and optims
		self.optimizer = torch.optim.Adam(self.hybrid.parameters(), lr=self.learningRate, weight_decay=1e-5)
		self.loss_func = nn.BCELoss()
		self.scheduler = StepLR(self.optimizer, step_size=5, gamma=0.1)

		self.qc_index = qc_index
		self.n_samples = n_samples


	def forward(self, x):
		x = self.cae.get_latent_space_state(x)
		x = self.hybrid(x)
		return x

	def predict(self, x):
		x = self.cae.get_latent_space_state(x)
		x = self.hybrid(x).tolist()[0]
		return 1 if x>0 else -1


	def trainADLoss(self, epochs=None, n_samples=None):
		"""
			The model is trained based on a mean 
			square loss.
		"""

		epochs = epochs if epochs else self.epochs
		n_samples = n_samples if n_samples else self.n_samples
		min_loss = 10
		best_params = self.hybrid.state_dict()
		scheduler = self.scheduler

		data_set = Variable(torch.FloatTensor(sample_test_data(n_samples)[0]))
		data_set = self.cae.get_latent_space_state(data_set).tolist()
		data_set = Variable(torch.FloatTensor(data_set))

		# TODO: Multi classs classification
		labels = np.array(sample_test_data(n_samples)[1])
		labels = np.where(labels==-1, 0, 1)
		labels = Variable(torch.FloatTensor(labels))

		if torch.cuda.is_available():
			data_set, labels = data_set.cuda(), labels.cuda()

		loss_list = []
		self.hybrid.train()
		for epoch in range(epochs):
			total_loss = []
			for i, x in enumerate(data_set):
				self.optimizer.zero_grad()
				output = (self.hybrid(x) + 1) / 2

				loss = self.loss_func(output, torch.FloatTensor([labels[i]]))
				loss.backward()
				self.optimizer.step()

				total_loss.append(loss.item())

			average_loss = sum(total_loss) / len(total_loss)
			loss_list.append(average_loss)
			print('Training [{:.0f}%]\tLoss: {:.4f}'.format(100. * (epoch + 1) / epochs, average_loss))

			if min_loss > average_loss:
				print("New min loss found!")
				best_params = self.hybrid.state_dict()
				min_loss = average_loss
			scheduler.step()

		path = f'../../data/training_results_ADLoss/pqc{self.qc_index}/training_result_loss_{round(min_loss, 3)}.pt'
		torch.save(best_params, os.path.join(dirname, path))

