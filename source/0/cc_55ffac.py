# https://github.com/JanusQ/AlgorithmBenchmark/blob/b729a8986d99bfe838b74ba34f213bfa4bb5e115/dataset/dataset2/cc.py
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import random

random.seed(555)


def gen_cc(qc, qr, nCoins):
    indexOfFalseCoin = random.randint(0, nCoins - 1)

    for i in range(nCoins):
        qc.h(qr[i])
    for i in range(nCoins):
        qc.cx(qr[i], qr[nCoins])
    qc.measure(qr[nCoins], cr[nCoins])

    qc.x(qr[nCoins]).c_if(cr, 0)
    qc.h(qr[nCoins]).c_if(cr, 0)

    for i in range(nCoins):
        qc.h(qr[i]).c_if(cr, 2 ** nCoins)
    qc.barrier()

    qc.cx(qr[indexOfFalseCoin], qr[nCoins]).c_if(cr, 0)
    qc.barrier()

    for i in range(nCoins):
        qc.h(qr[i]).c_if(cr, 0)

    # for i in range(nCoins):
    #     qc.measure(qr[i], cr[i])

def get_cir(nCoins):
    n_qubits = nCoins + 1
    qr = QuantumRegister(n_qubits)
    qc = QuantumCircuit(qr)

    gen_cc(qc, qr, cr, nCoins)
    return qc

print(get_cir(10))