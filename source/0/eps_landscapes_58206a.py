# https://github.com/obstjn/ws-qaoa-transfer/blob/841b427f1c71a4005db74d482bce1729fcbd8f7a/scripts/eps-landscapes.py
import numpy as np
import networkx as nx
from networkx import Graph
import matplotlib.pyplot as plt
from qiskit import Aer, execute
from qiskit.visualization import plot_histogram

from plotting import *
from calculations import get_energy_grid, maximizing_parameters
from circuit_generation import qaoa_circuit
from graph_management import *


G3reg1 = Graph()
G3reg1.add_edges_from([(0,1), (1,2), (1,3), (2,3), (3,4)])
#apx = np.array([1., .5, 1., 0, 0])
apx = np.array([1., 1., 0, 0, 0])
eps = np.array([0, .1, .2, .3, .4, .5])

#for e in eps:
#  # generate circuit
#  qc = qaoa_circuit(G3reg1, apx_sol=apx, eps=e)
#
#  # calculate & save energy
#  #energy = get_energy_grid(G3reg1, qc, edge=(1,3), samples=100)
#  #np.save(f'./ws-energies/3-reg_1/eps-energies/8_energy_eps_{e}.npy', energy)
#
#  # load energy
#  energy = np.load(f'./ws-energies/3-reg_1/eps-energies/8_energy_eps_{e}.npy')
#  
#  # plot ws landscape
#  plot_energy(energy, title=fr'$\varepsilon={e}$', filename=f'./ws-energies/3-reg_1/eps-energies/3reg1_8_eps_{e}')

#energy = get_energy_grid(G3reg1, qc, edge=(1,3), samples=100)
#np.save(f'./ws-energies/3-reg_1/eps-energies/23_energy_eps_{e}.npy', energy)

draw_graph_with_ws(G3reg1, apx)
