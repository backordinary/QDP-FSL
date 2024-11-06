# https://github.com/lukasszz/qiskit-exp/blob/ce14d53735870e7b6ace352629eb4049e9cd6740/emoticon.py
import qiskit
from PIL import Image
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.tools.visualization import matplotlib_circuit_drawer, plot_histogram

q = QuantumRegister(16)
c = ClassicalRegister(16)
qc = QuantumCircuit(q, c)

# 00101001
qc.x(q[0])
qc.x(q[3])
qc.x(q[5])

# second eight (qu)bits have superposition of
# '8' = 00111000
# ';' = 00111011
qc.h(q[9])  # create superposition on 9
qc.cx(q[9], q[8])  # spread it to 8 with a CNOT
qc.x(q[11])
qc.x(q[12])
qc.x(q[13])

# measure
for j in range(16):
    qc.measure(q[j], c[j])

qiskit.IBMQ.load_accounts()
backend = qiskit.IBMQ.get_backend('ibmq_qasm_simulator')

print("We'll use the least busy device:", backend.name())
shots=64
job = qiskit.execute(qc, backend, shots=shots)

stats = job.result().get_counts()
print(stats)
image = matplotlib_circuit_drawer(qc)

plot_histogram(stats)

import matplotlib.pyplot as plt

plt.rc('font', family='monospace')


def plot_smiley(stats, shots):
    for bitString in stats:
        char = chr(int(bitString[0:8],
                       2))  # get string of the leftmost 8 bits and convert to an ASCII character
        char += chr(int(bitString[8:16],
                        2))  # do the same for string of rightmost 8 bits, and add it to the previous character
        prob = stats[bitString] / shots  # fraction of shots for which this result occurred
        # create plot with all characters on top of each other with alpha given by how often it turned up in the output
        plt.annotate(char, (0.5, 0.5), va="center", ha="center", color=(0, 0, 0, prob), size=300)
        if (
                prob > 0.05):  # list prob and char for the dominant results (occurred for more than 5% of shots)
            print(str(prob) + "\t" + char)
    plt.axis('off')
    plt.show()


plot_smiley(stats, shots)
