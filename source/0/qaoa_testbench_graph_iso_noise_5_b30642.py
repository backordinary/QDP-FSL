# https://github.com/LegacYFTw/Graph-Isomorphism/blob/5a9333a6edba6461ea72f3b801e654c7f297cd8a/benchmark-data/qaoa_iso_pairs_edge_reduction/qaoa_testbench_graph_iso_noise_5.py
# -*- coding: utf-8 -*-
"""qaoa-testbench-graph-iso.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/LegacYFTw/Graph-Isomorphism/blob/main/notebooks/QAOA_Testbench.ipynb
"""

# !pip install 'qiskit[all]'
# !pip install matplotlib==3.1.3
# !pip install plotly
# !pip install gsgmorph

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from time import time
from pprint import pprint
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import clear_output

import itertools
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from dimod import BinaryQuadraticModel, AdjVectorBQM
from qiskit.optimization import QuadraticProgram
from qiskit.optimization.converters import LinearEqualityToPenalty, InequalityToEquality, IntegerToBinary
import neal
import pickle 
from qiskit.aqua.algorithms import QAOA
from qiskit.optimization.algorithms import MinimumEigenOptimizer
from qiskit import Aer
from qiskit import QuantumCircuit
from qiskit.visualization import plot_state_city
from qiskit.aqua.components.optimizers import COBYLA, SLSQP, ADAM, L_BFGS_B
from qiskit.optimization import QuadraticProgram
from docplex.mp.model import Model
import gsgmorph.matrix_form as gsgm_mf

from qiskit.optimization.applications.ising import stable_set
from qiskit import Aer
from qiskit.optimization.applications.ising import stable_set
from qiskit.aqua.algorithms import VQE, NumPyMinimumEigensolver, QAOA, NumPyEigensolver
from qiskit.aqua import aqua_globals
from qiskit.aqua import QuantumInstance
from qiskit.optimization.applications.ising.common import sample_most_likely
from qiskit.optimization.algorithms import MinimumEigenOptimizer

# Graphs are isomorphic. Dont get scared m8

with open(r"iso_pairs.pickle", "rb") as input_file:
  graph_data = pickle.load(input_file)

graph_data[1]

G1_data = graph_data[0][0]
G2_data = graph_data[0][1]

import networkx as nx
G1 = nx.Graph()
G2 = nx.Graph()
G1.add_edges_from(G1_data)
G2.add_edges_from(G2_data)

nx.draw(G1, with_labels=True)

nx.draw(G1, with_labels=True)

nx.is_isomorphic(G1,G2)

Q, sample_translation_dict = gsgm_mf.graph_isomorphism(G1, G2)

# Initialize a GPU backend
# Note that the cloud instance for tutorials does not have a GPU
# so this will raise an exception.
try:
    simulator_gpu = Aer.get_backend('aer_simulator')
    simulator_gpu.set_options(device='GPU')
except AerError as e:
    print(e)

def qubo_matrix_to_docplex(Q): 

  def build_matrix(data):
    data = dict(data)
    maxX = max([x for (x, y) in list(data.keys())])
    maxY = max([y for (x, y) in list(data.keys())])
    maxX = max([maxX, maxY])
    maxY = max([maxX, maxY])
    mat = np.zeros(shape=(maxX+1, maxY+1))
    for key, value in data.items():
        x, y = key
        mat[x][y] = value
    return mat

  # Print out the Qubo Matrix
  Q_matrix = build_matrix(Q)
  print(Q_matrix)
  print("Size of QUBO Matrix is: ", len(Q_matrix))

  def build_linear(Q, Q_matrix): 
    linear_dict = {}
    for idx in range(len(Q_matrix)): 
      linear_dict['x_{0}'.format(idx)] = Q[(idx,idx)]
    print('Linear part: ', linear_dict)
    return linear_dict

  def build_quadratic(Q, Q_matrix): 
    quadratic_dict = {}
    for node_pair in Q: 
      x,y = node_pair
      if x != y: 
        quadratic_dict[(f'x_{x}', f'x_{y}')] = Q[(x,y)]
    print('Quadratic Part: ', quadratic_dict)
    return quadratic_dict

  def build_constant(Q_matrix): 
    #Empirically decided!
    import math 
    number_of_nodes= math.sqrt(len(Q_matrix))
    constant = number_of_nodes * 2 - 4
    print(constant)
    return constant


  linear = build_linear(Q, Q_matrix)
  quadratic = build_quadratic(Q, Q_matrix)
  constant = build_constant(Q_matrix)

  mdl = QuadraticProgram('Graph Isomorphism')
  for node_pair in Q: 
    x,y = node_pair
    if x == y: 
      mdl.binary_var(name='x_{0}'.format(x))
  
  mdl.minimize(constant=constant, linear=linear, quadratic=quadratic)
  print(mdl.export_as_lp_string())

  return mdl

def qaoa_graph_isomorphism(graph_data, max_graph_pairs=2): 
  c = 0
  for graph_pair in graph_data:
    c = c + 1
    if (c > max_graph_pairs):
      break
    else: 
      G1_data = graph_pair[0]
      G2_data = graph_pair[1]
      G1 = nx.Graph()
      G2 = nx.Graph()
      G1.add_edges_from(G1_data)
      G2.add_edges_from(G2_data)
      print(nx.is_isomorphic(G1,G2))
      Q, sample_translation_dict = gsgm_mf.graph_isomorphism(G1, G2)
      mdl = qubo_matrix_to_docplex(Q)
      aqua_globals.random_seed = 10598
      quantum_instance = QuantumInstance(backend=simulator_gpu,
                                   seed_simulator=aqua_globals.random_seed,
                                   seed_transpiler=aqua_globals.random_seed)
      offset = 10  # Here we got the offset as 6
      trajectory={'beta_0':[], 'gamma_0':[], 'energy':[]}
      def callback(eval_count, params, mean, std_dev):
          trajectory['beta_0'].append(params[1])
          trajectory['gamma_0'].append(params[0])
          trajectory['energy'].append(-mean+offset)

      qaoa_mes = QAOA(quantum_instance=quantum_instance, initial_point=[0., 0.], callback=callback)
      exact_mes = NumPyMinimumEigensolver()
      qaoa = MinimumEigenOptimizer(qaoa_mes)
      # exact_result = exact.solve(mdl)
      # print(exact_result)
      qaoa_result = qaoa.solve(mdl)
      print(qaoa_result)
      savefile = "graph_5_nodes_edge_reduction_iso_results_pair_{}".format(c)
      with open(savefile, "wb") as output_file:
        pickle.dump({'qaoa_trajectory': trajectory, 'qaoa_object': qaoa_result, 'graph_1': G1_data, 'graph_2': G2_data, 'docplex_model': mdl}, output_file)

max_graph_pairs = 10
qaoa_graph_isomorphism(graph_data, max_graph_pairs=max_graph_pairs)

# from google.colab import files

# for i in range(max_graph_pairs):
#   filename = 'graph_4_nodes_iso_results_pair_{}'.format(i+1)
#   files.download(filename)
