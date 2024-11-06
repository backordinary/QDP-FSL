# https://github.com/lukasszz/qiskit-exp/blob/ce14d53735870e7b6ace352629eb4049e9cd6740/bitstring_enc/bitstring_encoding.py
# Cell
import math
import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer
from qiskit import execute
from qiskit.tools.visualization import plot_histogram

qiskit.IBMQ.load_accounts()
backend = Aer.get_backend('qasm_simulator')

shots = 100

# Cell

def encode_bitstring(bitstring, qr, cr, inverse=False):
    """
    create a circuit for constructing the quantum superposition of the bitstring
    """
    n = math.ceil(math.log2(len(bitstring))) + 1  # number of qubits
    assert n > 2, "the length of bitstring must be at least 2"

    qc = QuantumCircuit(qr, cr)

    # the probability amplitude of the desired state
    desired_vector = np.array([0.0 for i in range(2 ** n)])  # initialize to zero
    amplitude = np.sqrt(1.0 / 2 ** (n - 1))

    for i, b in enumerate(bitstring):
        pos = i * 2
        if b == "1" or b == "M":
            pos += 1
        desired_vector[pos] = amplitude

    if not inverse:
        # qc.initialize(desired_vector, [qr[i] for i in range(n)])
        qc.initialize(desired_vector, qr)
    else:
        qc.initialize(desired_vector, [qr[i] for i in range(n)]).inverse()  # invert the circuit

    return qc

# Cell
BS1 = '10000001'
BS2 = '11010011'

# Cell
n = math.ceil(math.log2(len(BS1))) + 1  # number of qubits
qr = QuantumRegister(n)
cr = ClassicalRegister(n)

qc_1 = encode_bitstring(BS1, qr, cr)
qc_1.measure(qr, cr)
qc_1.barrier()

job = execute(qc_1 , backend=backend, shots=shots)
st = job.result().get_counts()
print(st)
plot_histogram(st)

