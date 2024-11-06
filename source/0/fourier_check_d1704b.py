# https://github.com/rickapocalypse/final_paper_qiskit_sat/blob/bfd57cca11bdd3c70afb294bc74ed3e8ade27fa0/fourier_check.py
import qiskit.quantum_info as qi
from qiskit.circuit.library import FourierChecking
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

f = [0,0,0,0]
g = [1,1,1,-1]

circuit = FourierChecking(f=f, g=g)
circuit.draw(output='mpl')

zero = qi.Statevector.from_label('00')
sv = zero.evolve(circuit)
probs = sv.probabilities_dict()
plot_histogram(probs)
plt.show()