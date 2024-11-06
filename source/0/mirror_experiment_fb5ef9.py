# https://github.com/obstjn/ws-qaoa-transfer/blob/68d7085cf8ff593751ac7a13a713a73485dcf7fd/scripts/mirror_experiment.py
import numpy as np
import networkx as nx
from networkx import Graph
import matplotlib.pyplot as plt

from calculations import *
from circuit_generation import *
from plotting import *
from graph_management import *

from qiskit.visualization import plot_histogram
from qiskit import Aer, execute


G3reg0 = nx.read_adjlist('./graphs/3reg0.graph', nodetype=int)

qc0 = qaoa_circuit(G3reg0, apx_sol=[1, 0, 0, 0, 0, 0])
qc1 = qaoa_circuit(G3reg0, apx_sol=[1, 0, 0, 1, 0, 0])

#qc0.draw('mpl')
#qc1.draw('mpl')

gamma, beta = .4, .2 
beta_reflected = -beta + np.pi
gamma_reflected = -gamma + 2*np.pi
qaoa0 = qc0.assign_parameters([beta, gamma])
qaoa1 = qc1.assign_parameters([beta, gamma_reflected])
#qaoa0.draw('mpl')
#qaoa1.draw('mpl')
result0 = execute(qaoa0, Aer.get_backend('statevector_simulator'), shots=1).result()
result1 = execute(qaoa1, Aer.get_backend('statevector_simulator'), shots=1).result()
counts0 = result0.get_counts()
counts1 = result1.get_counts()

for key, prob in list(counts0.items()):
  if prob < 0.01:
    counts0.pop(key)
for key, prob in list(counts1.items()):
  if prob < 0.01:
    counts1.pop(key)

plot_histogram(counts0, sort='value', title=str(np.array([1,0,0,0,0,0])), figsize=(8,10))#, filename='histogram_0.pdf')
plot_histogram(counts1, sort='value', title=str(np.array([1,0,0,1,0,0])), figsize=(8,10))#, filename='histogram_11.pdf')

plt.show()
