# https://github.com/mohsenhariri/qisirq/blob/9b6e839365a87c099f85d8b97540951b5dedd63d/pkg/qiskit_ex2.py
import qiskit.quantum_info as qi
from qiskit.circuit.library import FourierChecking
from qiskit.visualization import plot_histogram

f = [1, -1, -1, -1]
g = [1, 1, -1, -1]

circ = FourierChecking(f=f, g=g)
print(circ.draw())
