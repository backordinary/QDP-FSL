# https://github.com/lukasszz/qiskit-exp/blob/ce14d53735870e7b6ace352629eb4049e9cd6740/bitstring_enc/string_compare_encode.py
import math
import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import execute
from qiskit.tools.visualization import plot_histogram

qiskit.IBMQ.load_accounts()
backend = qiskit.backends.ibmq.least_busy(qiskit.IBMQ.backends(simulator=True))
shots = 100

YEAST = "----------------------------------MM----------------------------"
PROTOZOAN = "--MM---------------M------------MMMM---------------M------------"
BACTERIAL = "---M---------------M------------MMMM---------------M------------"

YEAST = '10010010'
PROTOZOAN = '11010011'
BACTERIAL = '00000010'


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
        qc.initialize(desired_vector, [qr[i] for i in range(n)])
        qc.barrier(qr)
    else:
        qc.initialize(desired_vector, [qr[i] for i in range(n)]).inverse()  # invert the circuit
        for i in range(n):
            qc.measure(qr[i], cr[i])

    return qc


n = math.ceil(math.log2(len(YEAST))) + 1  # number of qubits
qr = QuantumRegister(n)
cr = ClassicalRegister(n)

qc_yeast = encode_bitstring(YEAST, qr, cr)
qc_yeast.measure(qr, cr)
job = execute(qc_yeast , backend=backend, shots=shots)
st = job.result().get_counts()
plot_histogram(st)

qc_yeast_inv = encode_bitstring(YEAST, qr, cr, inverse=True)
job = execute(qc_yeast_inv , backend=backend, shots=shots)
st = job.result().get_counts()
plot_histogram(st) # This gives info about how many times the specific state was mesured eg. state 0001.
# Do we still encode the states as the last register holds info about the data and the priouse ones about
# the position in the bitstring as it was mentioned in: "Comparing bitstrings of length 64 with 7 qubits"?

# Should we understand that state 000000 is the state when bitstring consist only with zeros?
qc_yeast = encode_bitstring(YEAST, qr, cr)
qc_yeast_inv = encode_bitstring(YEAST, qr, cr, inverse=True)
comb = qc_yeast + qc_yeast_inv
job = execute(comb , backend=backend, shots=shots)
st = job.result().get_counts()
plot_histogram(st) # 1.0 means that the bits are the same


qc_proto = encode_bitstring(PROTOZOAN, qr, cr)
qc_yeast_inv = encode_bitstring(YEAST, qr, cr, inverse=True)
comb = qc_proto + qc_yeast_inv
job = execute(comb , backend=backend, shots=shots)
st = job.result().get_counts()
plot_histogram(st) #