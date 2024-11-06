# https://github.com/mar-ven/iquhack/blob/377899d16122ad707ca53980f93a572a68783ed5/hackaton/2023_IonQ_Remote1/part1.py
import cirq
import qiskit
from qiskit.circuit.library import RealAmplitudes
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import numpy as np
from qiskit_machine_learning.circuit.library import RawFeatureVector

def ansatz(num_qubits):
    return RealAmplitudes(num_qubits, reps=5)

# Maybe we dont need this
# def encode_cirq(image):
#     circuit=cirq.Circuit()
#     if image[0][0]==0:
#         circuit.append(cirq.rx(np.pi).on(cirq.LineQubit(0)))
#     return circuit

def encode_qiskit(image):
    num_latent = 3
    num_trash = 2

    fm = RawFeatureVector(2 ** (num_latent + num_trash))
    #fm = fm.assign_parameters(image)
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    ae = QuantumCircuit(qr, cr)
    ae.compose(ansatz(num_latent + num_trash), range(0, num_latent + num_trash), inplace=True)
    qc = QuantumCircuit(num_latent + 2 * num_trash + 1, 1)
    qc = qc.compose(fm, range(num_latent + num_trash))
    qc = qc.compose(ae)
    qc.draw("mpl")
    
    return qc


def decode(histogram):
    if 1 in histogram.keys():
        image=[[0,0],[0,0]]
    else:
        image=[[1,1],[1,1]]
    return image