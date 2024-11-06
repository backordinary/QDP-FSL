# https://github.com/Gitiauxx/QuantDist/blob/ec9c7f6030bcd027b662822d0df845ea7c8f01cc/examples/four_qubits_real.py
import numpy as np
from  math import sqrt
import matplotlib.pyplot as plt

from qiskit import IBMQ
#IBMQ.save_account('d5c3abdfb2d464260eeda4fc0aecd1ed1e3e64c270703dd5a0a34a1546f57c7d7a1e0d19d695c0adc43f509b7219cff4f86da11e818edcf095dfab7403649c1f')
IBMQ.load_account()

from qiskit import QuantumCircuit, QuantumRegister, execute, Aer, ClassicalRegister, BasicAer
from qiskit.compiler import transpile, assemble
from qiskit.visualization import plot_gate_map

from source.utils import get_logger

logger = get_logger(__name__)

r0 = [1, 0, 0, 0]
r1 = [sqrt(5 / 9), sqrt(4 / 9), 0, 0]
r2 = [sqrt(5 / 9), -sqrt(1 / 9), 0. + 1j * sqrt(1 / 3), 0]
r3 = [sqrt(5 / 9), -sqrt(1 / 9), 0. - 1j * sqrt(1 / 3), 0]

initial_state_list = [r0, r1, r2, r3]

qr = QuantumRegister(8, 'q')
anc = QuantumRegister(5, 'ancilla')
cr = ClassicalRegister(5, 'c')
circuit = QuantumCircuit(anc, qr, cr)

for i, r in enumerate(initial_state_list):
    circuit.initialize(r, [qr[2 * i], qr[2 * i + 1]])
circuit.h(anc[0])
circuit.h(anc[1])
circuit.h(anc[2])
circuit.h(anc[3])
circuit.h(anc[4])
circuit.cswap(anc[4], qr[0], qr[2 + 0])
circuit.cswap(anc[4], qr[4 + 0], qr[6 + 0])
circuit.cswap(anc[3], qr[0], qr[4 + 0])
circuit.cswap(anc[2], qr[2 + 0], qr[6 + 0])
circuit.cswap(anc[1], qr[2 + 0], qr[4 + 0])

circuit.cswap(anc[0], qr[0], qr[2 + 0])
circuit.cswap(anc[4], qr[1], qr[2 + 1])
circuit.cswap(anc[4], qr[4 + 1], qr[6 + 1])
circuit.cswap(anc[3], qr[1], qr[4 + 1])
circuit.cswap(anc[2], qr[2 + 1], qr[6 + 1])
circuit.cswap(anc[1], qr[2 + 1], qr[4 + 1])

circuit.cswap(anc[0], qr[1], qr[2 + 1])

circuit.h(anc[0])

circuit.measure(anc, cr)
circuit.draw(output='mpl')
plt.show()

# Execute the circuit on the qasm simulator
provider = IBMQ.get_provider(hub='ibm-q')
backend = provider.get_backend('ibmq_16_melbourne')

mapped_circuit = transpile(circuit, backend=backend)

qobj = assemble(mapped_circuit, backend=backend, shots=8192)
job = backend.run(qobj)
result = job.result()

logger.info(job.status)

# Returns counts
counts = result.get_counts(circuit)
total = sum(counts.values())
counts_final = {state[:-1]: c for state, c in counts.items() if state[-1] == '0'}

counts = {state: c / total for state, c in counts_final.items()}

r12 = counts['0000'] * 2 ** 5 - 1
r13 = counts['0001'] * 2 ** 5 - 1
r14 = counts['0010'] * 2 ** 5 - 1
r23 = counts['1010'] * 2 ** 5 - 1
r24 = counts['1001'] * 2 ** 5 - 1
r34 = counts['0110'] * 2 ** 5 - 1


logger.info(f'LHS of face is {r12 + r13 + r14 - r23 - r24 - r34}')



