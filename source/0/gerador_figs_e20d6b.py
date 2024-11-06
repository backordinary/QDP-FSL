# https://github.com/rickapocalypse/final_paper_qiskit_sat/blob/bfd57cca11bdd3c70afb294bc74ed3e8ade27fa0/gerador_figs.py
from qiskit.providers.aer import QasmSimulator
from multiprocessing import Barrier
from qiskit import*
from qiskit import circuit
import matplotlib.pyplot as plt
from qiskit.tools.visualization import plot_histogram
from qiskit.tools.monitor import job_monitor

circuit = QuantumCircuit(2)
circuit.cx(0,1)
circuit.draw(output='mpl')
plt.show()