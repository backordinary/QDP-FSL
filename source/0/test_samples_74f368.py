# https://github.com/soosub/bachelor-thesis/blob/7da3447e1fd77a9d94f79b7939dfc952d4f0e11e/Implementation/Tests/test_samples.py
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 13:01:26 2020

@author: joost
"""
from QAOA import QAOA
import my_graphs
import qiskit


G = my_graphs.cycle_graph(10)
backend = qiskit.Aer.get_backend('qasm_simulator')

# These were the angles found using pyQuil, checking if my angle definitions are consistent with theirs
gamma = [2.60900986, 2.23741483, 2.09456714, 2.03917042, 2.01950908, 1.99870163,
     1.96350832, 1.93678678, 1.89051587, 1.84553579, 1.80890563, 1.80109892,
     1.80292777, 1.84364326, 1.88012771, 1.91741207, 1.94778564, 1.9756463,
     1.97435811, 1.88903855]
beta = [2.42069956, 2.47326332, 2.50991822, 2.53493295, 2.51952154, 2.52032868,
 2.45863663, 2.4596706,  2.45556628, 2.44420141, 2.42869925, 2.44304717,
 2.45733293, 2.47683847, 2.53614331, 2.60218736, 2.68949007, 2.75713179,
 2.84788729, 2.9904129 ]

stats = QAOA.sample(G, gamma, beta, backend, n_samples = 1024, plot_histogram = True)

print(max(stats, key=stats.get))