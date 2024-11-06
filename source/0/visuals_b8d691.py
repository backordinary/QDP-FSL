# https://github.com/cpbunker/learn_qiskit/blob/08911639eb8fd94b2c5101ccfb24e6385cfc9a27/intro/visuals.py
'''
https://github.com/cpbunker/learn/qiskit
'''

import qiskit
from qiskit.visualization import plot_bloch_vector

import matplotlib.pyplot as plt
plt.rcParams.update({"text.usetex": True,"font.family": "Times"})

fig = plot_bloch_vector([0,1,0]);
plt.show();
