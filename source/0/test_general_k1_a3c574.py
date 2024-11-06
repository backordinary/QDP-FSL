# https://github.com/tigerjack/qiskit-tests/blob/b22361a5fbce84639a0019fa32edfba452ef8384/src/w_states/test_general_k1.py
from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister
from qiskit import execute, Aer
import qiskit.extensions.simulator
import numpy as np
from math import sqrt
import qiskit_support as qsp

def w1(qc, qr, theta,stopAt=0):
    snapshot_idx = 0
    qc.u3(theta[0], 0, 0, qr[0])
    qc.snapshot(str(snapshot_idx))
    snapshot_idx+=1
    for i in range(1, len(theta)):
        print(f"theta{theta[i]} qr {i-1}, qr {i}")
        qc.cu3(theta[i], 0, 0, qr[i - 1], qr[i])
        qc.snapshot(str(snapshot_idx))
        snapshot_idx+=1
    for i in reversed(range(1, len(qr))):
        print(f"cx {i-1} {i}")
        qc.cx(qr[i - 1], qr[i])
        qc.snapshot(str(snapshot_idx))
        snapshot_idx+=1
    qc.x(qr[0])
    qc.snapshot(str(snapshot_idx))
    snapshot_idx+=1
    return snapshot_idx

def main():
    n = 4
    arccos = np.arccos
    theta = []
    for i in range(n - 1, 0, -1):
        print(1, i)
        theta.append(2 * arccos(sqrt(1 / (1 + i))))
    print(theta)
    qr = QuantumRegister(n)
    cr = ClassicalRegister(len(qr))
    qc = QuantumCircuit(qr, cr)
    snapshot_idx = w1(qc, qr, theta)
    qc.barrier()
    # qc += qc.inverse()
    statevector_and_snapshots(qc, snapshot_idx)
    print(qc)
    qc.measure(qr, cr)
    qasm(qc)

def qasm(qc):
    b = Aer.get_backend('qasm_simulator')
    result = execute(qc, b, shots=1000).result()
    # d = {k:v for }
    print(sorted(result.get_counts().items()))

def statevector_and_snapshots(qc, snapshot_idx):
    b = Aer.get_backend('statevector_simulator')
    result = execute(qc, b).result()
    print(result.get_statevector())
    print("")
    for i in range(snapshot_idx):
        sv = result.data()['snapshots']['statevector'][str(i)].pop()
        print(sv)
        print("")
        # dets = qsp.from_statevector_to_prob_and_phase_detailed(sv, qc)
        # print(dets)

if __name__ == '__main__':
    main()
