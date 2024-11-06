# https://github.com/PedruuH/Computacao-Quantica/blob/c39194368dbb02ebbafc9858904bad958f648e94/teste.py


import math
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit import execute, BasicAer

from qiskit.tools.visualization import plot_histogram 


###############################################################
# make the qft
###############################################################
def input_state(circ, n):
    """n-qubit input state for QFT that produces output 1."""
    for j in range(n):
        circ.h(j)
        circ.p(-math.pi / float(2 ** (j)), j)


def qft(circ, n):
    """n-qubit QFT on q in circ."""
    for j in range(n):
        for k in range(j):
            circ.cp(math.pi / float(2 ** (j - k)), j, k)
        circ.h(j)


qft1 = QuantumCircuit(5, 5, name="qft")


input_state(qft1, 3)
qft1.barrier()
qft(qft1, 3)
qft1.barrier()
qft1.measure(0, 0)
qft1.measure(1, 1)
qft1.measure(2, 2)

print(qft1)


sim_backend = BasicAer.get_backend("qasm_simulator")
job = execute([qft1], sim_backend, shots=1024)
result = job.result()
qft1.draw(output='mpl')
plt.show()
plot_histogram(result.get_counts(qft1))
plt.show()
