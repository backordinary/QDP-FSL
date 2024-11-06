# https://github.com/soosub/bachelor-thesis/blob/7da3447e1fd77a9d94f79b7939dfc952d4f0e11e/Implementation/Tests/test_ibmq.py
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 13:38:00 2020

@author: joost
"""

from qiskit import IBMQ
from QAOA import QAOA
import my_graphs

G = my_graphs.diamond()
p = 1
gamma, beta = [0], [0]

provider = IBMQ.load_account()
backend = provider.get_backend('ibmq_qasm_simulator')

QAOA.sample(G, gamma, beta, backend, n_samples = 1024, plot_histogram = False)