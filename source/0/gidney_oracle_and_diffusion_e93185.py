# https://github.com/Sammyalhashe/Thesis/blob/c22cff964f1c635eb28be1130c02fe2d95e536c8/Grover/Gidney/Deprecated/gidney_oracle_and_diffusion.py
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import execute  # available_backends
# from qiskit.tools.visualization import plot_state, plot_histogram
from qiskit import Aer  # IBMQ
# import Qconfig
from qiskit.tools.visualization import circuit_drawer
# from matplotlib import pyplot as plt
# import numpy as np
# from scipy import linalg as la
from math import sqrt


import sys
if sys.version_info < (3, 5):
    raise Exception('Run with python 3')

# oracle definition

# number of bits denoting the index of the index
index = '1110'


def Grover(index):
    n = len(index)
    if not n > 2:
        raise ValueError
    # Create a Quantum Register with 2 qubits.
    # control quibits
    q = QuantumRegister(n, 'q')
    # ancillary qubits
    anc = QuantumRegister(n - 1, 'anc')
    # control for control z-gate in oracle
    # target qubits for oracle (for debuggin)
    tar = QuantumRegister(1, 'tar')
    # ancillary qubits for diffusion gate
    ancD = QuantumRegister(n-2, 'ancD')
    # target qubits for diffusion gate
    tarD = QuantumRegister(1, 'tarD')
    # Create a Classical Register for oracle measurements.
    # cOracle = ClassicalRegister(1, 'cOracle')
    # registers for measurement of the qubits
    c = ClassicalRegister(n, 'c')
    # Create a Quantum Circuit
    qc = QuantumCircuit(q, anc, tar, ancD, tarD, c)

    # # set input for testing
    # for i in range(n):
    #     v = int(index[i])
    #     if v != 0:
    #         qc.x(q[i])

    # apply hadamard gates
    for i in range(n):
        qc.h(q[i])

    # qc.x(tar[0])
    #################################################
    """ORACLE IMPLEMENTATION
    """
    def oracle1():
        # applying control-NOT gates forward
        for i in range(n):
            v = int(index[i])
            # print(v)
            if i == 0:
                v2 = int(index[i+1])
                if v2 == 0:
                    qc.x(q[i + 1])
                if v == 0:
                    qc.x(q[i])
                    qc.ccx(q[i], q[i + 1], anc[i])
                    # qc.x(q[i])
                else:
                    qc.ccx(q[i], q[i + 1], anc[i])
                # if v2 == 0:
                #     qc.x(q[i + 1])
            elif i == 1:
                pass

            else:
                if v == 0:
                    qc.x(q[i])
                    qc.ccx(q[i], anc[i - 2], anc[i - 1])
                    # qc.x(q[i])
                else:
                    qc.ccx(q[i], anc[i - 2], anc[i - 1])
        qc.cx(anc[n - 2], tar[0])
        qc.z(tar[0])
        # qc.measure(tar, cOracle)

        qc.cx(anc[n - 2], tar[0])
        # applying control-NOT gates backwards -> reset
        for i in range(n-1, -1, -1):
            v = int(index[i])
            # print(v)
            if i == 0:
                v2 = int(index[i+1])
                # if v2 == 0:
                #     qc.x(q[i + 1])
                if v == 0:
                    # qc.x(q[i])
                    qc.ccx(q[i], q[i + 1], anc[i])
                    qc.x(q[i])
                else:
                    qc.ccx(q[i], q[i + 1], anc[i])
                if v2 == 0:
                    qc.x(q[i + 1])
            elif i == 1:
                pass

            else:
                if v == 0:
                    # qc.x(q[i])
                    qc.ccx(q[i], anc[i - 2], anc[i - 1])
                    qc.x(q[i])
                else:
                    qc.ccx(q[i], anc[i - 2], anc[i - 1])

    #################################################
    """GROVER-DIFFUSION GATE IMPLEMENTAION
    """
    def diffusion_gate():
        # apply hadamard gates
        for i in range(n):
            qc.h(q[i])

        # apply pauli-X gates
        for i in range(n):
            qc.x(q[i])

        """Apply multi-qubit control-pauli-Z gate
        """
        for i in range(n - 1):
            if i == 0:
                qc.ccx(q[i], q[i + 1], ancD[i])
            elif i == 1:
                pass
            else:
                qc.ccx(q[i], ancD[i - 2], ancD[i - 1])
        qc.cz(ancD[n-3], q[n-1])

        for i in range(n - 1):
            if i == 0:
                qc.ccx(q[i], q[i + 1], ancD[i])
            elif i == 1:
                pass
            else:
                qc.ccx(q[i], ancD[i - 2], ancD[i - 1])

        # apply pauli-X gates
        for i in range(n):
            qc.x(q[i])

        # apply hadamard gates
        for i in range(n):
            qc.h(q[i])

    #################################################

    """Grover implementation repeating oracle + diffusion
    """
    N = int(sqrt(n))

    for i in range(N):
        oracle1()
        diffusion_gate()
    #################################################
    """Measurement of quibits
    """

    for i in range(n):
        qc.measure(q[i], c[i])

    #################################################
    circuit_drawer(qc, filename='gidney2.png')

    # See a list of available local simulators
    print("Local backends: ", Aer.available_backends())

    # compile and run the Quantum circuit on a simulator backend
    backend_sim = Aer.get_backend('qasm_simulator')
    job_sim = execute(qc, backend_sim)
    result_sim = job_sim.result()

    # Show the results
    print("simulation: ", result_sim)
    print(result_sim.get_counts(qc))


if __name__ == '__main__':
    Grover(index)
