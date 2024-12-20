# https://github.com/daglarcoban/distributed-quantum-computing/blob/11b971ffd98c9035cc0e4345941ed47ac5d4c121/src/shor_code_variants/n_clusters_of_n/shor_code_4_clusters_of_4.py
import numpy as np

from qiskit.circuit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import execute

from quantuminspire.qiskit import QI

from src.util.authentication import QI_authenticate

from src.util.cat_disentangler import get_cat_disentangler_circuit
from src.util.cat_entangler import get_cat_entangler_circuit

def get_shor_code_4_c_4(error_cluster=None, error_type=None, error_bit=None, a = None, b = None):
    # Last (qu)bit in each register is the channel one per cluster
    c_a = [ClassicalRegister(1) for _ in range(5)]
    c_b = [ClassicalRegister(1) for _ in range(5)]
    c_c = [ClassicalRegister(1) for _ in range(5)]
    c_d = [ClassicalRegister(1) for _ in range(5)]
    q_a = QuantumRegister(5)
    q_b = QuantumRegister(5)
    q_c = QuantumRegister(5)
    q_d = QuantumRegister(5)
    q = [q_a, q_b, q_c, q_d]

    circuit_a = QuantumCircuit(q_a)
    for reg in c_a:
        circuit_a.add_register(reg)
    circuit_b = QuantumCircuit(q_b)
    for reg in c_b:
        circuit_b.add_register(reg)
    circuit_c = QuantumCircuit(q_c)
    for reg in c_c:
        circuit_c.add_register(reg)
    circuit_d = QuantumCircuit(q_d)
    for reg in c_d:
        circuit_d.add_register(reg)


    # Initialize the main qubit that will be error corrected
    alpha = 0  # 1 / sqrt(2)
    if a is not None:
        alpha = a
    beta = 1  # / sqrt(2)
    if b is not None:
        beta = b

    circuit_a.initialize([alpha, beta], q_a[0])

    circuit = circuit_a + circuit_b + circuit_c + circuit_d

    #First part of the phase flip code
    # cnot with cluster a and b
    circuit = circuit.compose(get_cat_entangler_circuit(2), [q_a[0], q_a[4], q_b[4]], [c_a[0][0], c_a[4][0], c_b[4][0]])
    circuit.cx(q_b[4], q_b[0])
    circuit = circuit.compose(get_cat_disentangler_circuit(2), [q_a[0], q_a[4], q_b[4]], [c_a[0][0], c_a[4][0], c_b[4][0]])

    circuit.barrier()

    # cnot with cluster a and c
    circuit = circuit.compose(get_cat_entangler_circuit(2), [q_a[0], q_a[4], q_c[4]], [c_a[0][0], c_a[4][0], c_c[4][0]])
    circuit.cx(q_c[4], q_c[0])
    circuit = circuit.compose(get_cat_disentangler_circuit(2), [q_a[0], q_a[4], q_c[4]], [c_a[0][0], c_a[4][0], c_c[4][0]])

    circuit.barrier()

    # cnot with cluster a and d
    circuit = circuit.compose(get_cat_entangler_circuit(2), [q_a[0], q_a[4], q_d[4]], [c_a[0][0], c_a[4][0], c_d[4][0]])
    circuit.cx(q_d[4], q_d[0])
    circuit = circuit.compose(get_cat_disentangler_circuit(2), [q_a[0], q_a[4], q_d[4]], [c_a[0][0], c_a[4][0], c_d[4][0]])

    circuit.barrier()

    circuit.h(q_a[0])
    circuit.h(q_b[0])
    circuit.h(q_c[0])
    circuit.h(q_d[0])

    circuit.cx(q_a[0], q_a[1])
    circuit.cx(q_a[0], q_a[2])
    circuit.cx(q_a[0], q_a[3])
    circuit.cx(q_b[0], q_b[1])
    circuit.cx(q_b[0], q_b[2])
    circuit.cx(q_b[0], q_b[3])
    circuit.cx(q_c[0], q_c[1])
    circuit.cx(q_c[0], q_c[2])
    circuit.cx(q_c[0], q_c[3])

    circuit.barrier()  # until ERROR BLOCK

    ### ERROR BLOCK --- START
    if error_type == 'random':
        RNG = np.random.random(1)
        if RNG >= 0.66:
            error_type = 'z'
        elif RNG >= 0.33 and RNG < 0.66:
            error_type = 'x'
        else:
            error_type = 'y'
    if error_bit == 'random':
        error_bit = np.random.choice([0, 1, 2, 3])
    if error_cluster == 'random':
        error_cluster = np.random.choice([0, 1, 2, 3])

    if error_bit != None:
        if error_type == 'z':
            circuit.z(q[error_cluster][error_bit])
        elif error_type == 'x':
            circuit.x(q[error_cluster][error_bit])
        elif error_type == 'y':
            circuit.y(q[error_cluster][error_bit])

    ## ERROR BLOCK --- END

    circuit.barrier()  # after ERROR BLOCK
    circuit.cx(q_a[0], q_a[1])
    circuit.cx(q_a[0], q_a[2])
    circuit.cx(q_a[0], q_a[3])
    circuit.cx(q_b[0], q_b[1])
    circuit.cx(q_b[0], q_b[2])
    circuit.cx(q_b[0], q_b[3])
    circuit.cx(q_c[0], q_c[1])
    circuit.cx(q_c[0], q_c[2])
    circuit.cx(q_c[0], q_c[3])

    #LOCAL CCCNOT GATES START
    circuit.mct([q_a[1], q_a[2], q_a[3]], q_a[0], None, mode="advanced")
    circuit.mct([q_b[1], q_b[2], q_b[3]], q_b[0], None, mode="advanced")
    circuit.mct([q_c[1], q_c[2], q_c[3]], q_c[0], None, mode="advanced")
    circuit.mct([q_d[1], q_d[2], q_d[3]], q_d[0], None, mode="advanced")

    #Splitting up the CCCNOT gates for calculating circuit depth adequately
    # CCCNOT WITHIN CLUSTER A
    # toffoli(circuit, 3, 2, 1, q_a)
    # circuit.h(q_a[0])
    # circuit.tdg(q_a[1])
    # circuit.tdg(q_a[0])
    # circuit.cx(q_a[1], q_a[0])
    # circuit.t(q_a[0])
    # circuit.cx(q_a[1], q_a[0])
    # toffoli(circuit, 3, 2, 1, q_a)
    # circuit.t(q_a[1])
    # circuit.t(q_a[0])
    # circuit.cx(q_a[1], q_a[0])
    # circuit.tdg(q_a[0])
    # circuit.cx(q_a[1], q_a[0])
    # circuit.cx(q_a[3], q_a[2])
    # circuit.rz(-1 / 8 * np.pi, q_a[2])
    # circuit.rz(-1 / 8 * np.pi, q_a[0])
    # circuit.cx(q_a[2], q_a[0])
    # circuit.rz(1 / 8 * np.pi, q_a[0])
    # circuit.cx(q_a[2], q_a[0])
    # circuit.cx(q_a[3], q_a[2])
    # circuit.rz(1 / 8 * np.pi, q_a[2])
    # circuit.rz(1 / 8 * np.pi, q_a[0])
    # circuit.cx(q_a[2], q_a[0])
    # circuit.rz(-1 / 8 * np.pi, q_a[0])
    # circuit.cx(q_a[2], q_a[0])
    # circuit.rz(1 / 8 * np.pi, q_a[3])
    # circuit.rz(1 / 8 * np.pi, q_a[0])
    # circuit.cx(q_a[3], q_a[0])
    # circuit.rz(-1 / 8 * np.pi, q_a[0])
    # circuit.cx(q_a[3], q_a[0])
    # circuit.h(q_a[0])
    #
    # # CCCNOT WITHIN CLUSTER B
    # toffoli(circuit, 3, 2, 1, q_b)
    # circuit.h(q_b[0])
    # circuit.tdg(q_b[1])
    # circuit.tdg(q_b[0])
    # circuit.cx(q_b[1], q_b[0])
    # circuit.t(q_b[0])
    # circuit.cx(q_b[1], q_b[0])
    # toffoli(circuit, 3, 2, 1, q_b)
    # circuit.t(q_b[1])
    # circuit.t(q_b[0])
    # circuit.cx(q_b[1], q_b[0])
    # circuit.tdg(q_b[0])
    # circuit.cx(q_b[1], q_b[0])
    # circuit.cx(q_b[3], q_b[2])
    # circuit.rz(-1 / 8 * np.pi, q_b[2])
    # circuit.rz(-1 / 8 * np.pi, q_b[0])
    # circuit.cx(q_b[2], q_b[0])
    # circuit.rz(1 / 8 * np.pi, q_b[0])
    # circuit.cx(q_b[2], q_b[0])
    # circuit.cx(q_b[3], q_b[2])
    # circuit.rz(1 / 8 * np.pi, q_b[2])
    # circuit.rz(1 / 8 * np.pi, q_b[0])
    # circuit.cx(q_b[2], q_b[0])
    # circuit.rz(-1 / 8 * np.pi, q_b[0])
    # circuit.cx(q_b[2], q_b[0])
    # circuit.rz(1 / 8 * np.pi, q_b[3])
    # circuit.rz(1 / 8 * np.pi, q_b[0])
    # circuit.cx(q_b[3], q_b[0])
    # circuit.rz(-1 / 8 * np.pi, q_b[0])
    # circuit.cx(q_b[3], q_b[0])
    # circuit.h(q_b[0])
    #
    # # CCCNOT WITHIN CLUSTER C
    # toffoli(circuit, 3, 2, 1, q_c)
    # circuit.h(q_c[0])
    # circuit.tdg(q_c[1])
    # circuit.tdg(q_c[0])
    # circuit.cx(q_c[1], q_c[0])
    # circuit.t(q_c[0])
    # circuit.cx(q_c[1], q_c[0])
    # toffoli(circuit, 3, 2, 1, q_c)
    # circuit.t(q_c[1])
    # circuit.t(q_c[0])
    # circuit.cx(q_c[1], q_c[0])
    # circuit.tdg(q_c[0])
    # circuit.cx(q_c[1], q_c[0])
    # circuit.cx(q_c[3], q_c[2])
    # circuit.rz(-1 / 8 * np.pi, q_c[2])
    # circuit.rz(-1 / 8 * np.pi, q_c[0])
    # circuit.cx(q_c[2], q_c[0])
    # circuit.rz(1 / 8 * np.pi, q_c[0])
    # circuit.cx(q_c[2], q_c[0])
    # circuit.cx(q_c[3], q_c[2])
    # circuit.rz(1 / 8 * np.pi, q_c[2])
    # circuit.rz(1 / 8 * np.pi, q_c[0])
    # circuit.cx(q_c[2], q_c[0])
    # circuit.rz(-1 / 8 * np.pi, q_c[0])
    # circuit.cx(q_c[2], q_c[0])
    # circuit.rz(1 / 8 * np.pi, q_c[3])
    # circuit.rz(1 / 8 * np.pi, q_c[0])
    # circuit.cx(q_c[3], q_c[0])
    # circuit.rz(-1 / 8 * np.pi, q_c[0])
    # circuit.cx(q_c[3], q_c[0])
    # circuit.h(q_c[0])
    #
    # # CCCNOT WITHIN CLUSTER D
    # toffoli(circuit, 3, 2, 1, q_d)
    # circuit.h(q_d[0])
    # circuit.tdg(q_d[1])
    # circuit.tdg(q_d[0])
    # circuit.cx(q_d[1], q_d[0])
    # circuit.t(q_d[0])
    # circuit.cx(q_d[1], q_d[0])
    # toffoli(circuit, 3, 2, 1, q_d)
    # circuit.t(q_d[1])
    # circuit.t(q_d[0])
    # circuit.cx(q_d[1], q_d[0])
    # circuit.tdg(q_d[0])
    # circuit.cx(q_d[1], q_d[0])
    # circuit.cx(q_d[3], q_d[2])
    # circuit.rz(-1 / 8 * np.pi, q_d[2])
    # circuit.rz(-1 / 8 * np.pi, q_d[0])
    # circuit.cx(q_d[2], q_d[0])
    # circuit.rz(1 / 8 * np.pi, q_d[0])
    # circuit.cx(q_d[2], q_d[0])
    # circuit.cx(q_d[3], q_d[2])
    # circuit.rz(1 / 8 * np.pi, q_d[2])
    # circuit.rz(1 / 8 * np.pi, q_d[0])
    # circuit.cx(q_d[2], q_d[0])
    # circuit.rz(-1 / 8 * np.pi, q_d[0])
    # circuit.cx(q_c[2], q_d[0])
    # circuit.rz(1 / 8 * np.pi, q_d[3])
    # circuit.rz(1 / 8 * np.pi, q_d[0])
    # circuit.cx(q_d[3], q_d[0])
    # circuit.rz(-1 / 8 * np.pi, q_d[0])
    # circuit.cx(q_d[3], q_d[0])
    # circuit.h(q_d[0])

    #LOCAL CCCNOT GATES END

    circuit.barrier()  # until h gates

    circuit.h(q_a[0])
    circuit.h(q_b[0])
    circuit.h(q_c[0])
    circuit.h(q_d[0])

    circuit.barrier()

    #NON LOCAL CNOT STUFF

    # cnot with cluster a and b
    circuit = circuit.compose(get_cat_entangler_circuit(2), [q_a[0], q_a[4], q_b[4]], [c_a[0][0], c_a[4][0], c_b[4][0]])
    circuit.cx(q_b[4], q_b[0])
    circuit = circuit.compose(get_cat_disentangler_circuit(2), [q_a[0], q_a[4], q_b[4]], [c_a[0][0], c_a[4][0], c_b[4][0]])

    circuit.barrier()

    # cnot with cluster a and c
    circuit = circuit.compose(get_cat_entangler_circuit(2), [q_a[0], q_a[4], q_c[4]], [c_a[0][0], c_a[4][0], c_c[4][0]])
    circuit.cx(q_c[4], q_c[0])
    circuit = circuit.compose(get_cat_disentangler_circuit(2), [q_a[0], q_a[4], q_c[4]], [c_a[0][0], c_a[4][0], c_c[4][0]])

    circuit.barrier()

    # cnot with cluster a and d
    circuit = circuit.compose(get_cat_entangler_circuit(2), [q_a[0], q_a[4], q_d[4]], [c_a[0][0], c_a[4][0], c_d[4][0]])
    circuit.cx(q_d[4], q_d[0])
    circuit = circuit.compose(get_cat_disentangler_circuit(2), [q_a[0], q_a[4], q_d[4]], [c_a[0][0], c_a[4][0], c_d[4][0]])

    circuit.barrier()

    ### NON LOCAL CCCX GATES --- START

    ##FIRST NORMAL TOFFOLI
    circuit.h(q_b[0])

    circuit = circuit.compose(get_cat_entangler_circuit(2), [q_c[0], q_c[4], q_b[4]], [c_c[0][0], c_c[4][0], c_b[4][0]])
    circuit.cx(q_b[4], q_b[0])
    circuit = circuit.compose(get_cat_disentangler_circuit(2), [q_c[0], q_c[4], q_b[4]], [c_c[0][0], c_c[4][0], c_b[4][0]])

    circuit.tdg(q_b[0])

    circuit = circuit.compose(get_cat_entangler_circuit(2), [q_d[0], q_d[4], q_b[4]], [c_d[0][0], c_d[4][0], c_b[4][0]])
    circuit.cx(q_b[4], q_b[0])
    circuit = circuit.compose(get_cat_disentangler_circuit(2), [q_d[0], q_d[4], q_b[4]], [c_d[0][0], c_d[4][0], c_b[4][0]])

    circuit.t(q_b[0])

    circuit = circuit.compose(get_cat_entangler_circuit(2), [q_c[0], q_c[4], q_b[4]], [c_c[0][0], c_c[4][0], c_b[4][0]])
    circuit.cx(q_b[4], q_b[0])
    circuit = circuit.compose(get_cat_disentangler_circuit(2), [q_c[0], q_c[4], q_b[4]], [c_c[0][0], c_c[4][0], c_b[4][0]])

    circuit.tdg(q_b[0])

    circuit = circuit.compose(get_cat_entangler_circuit(2), [q_d[0], q_d[4], q_b[4]], [c_d[0][0], c_d[4][0], c_b[4][0]])
    circuit.cx(q_b[4], q_b[0])
    circuit = circuit.compose(get_cat_disentangler_circuit(2), [q_d[0], q_d[4], q_b[4]], [c_d[0][0], c_d[4][0], c_b[4][0]])

    circuit.t(q_b[0])
    circuit.t(q_c[0])

    circuit = circuit.compose(get_cat_entangler_circuit(2), [q_d[0], q_d[4], q_c[4]], [c_d[0][0], c_d[4][0], c_c[4][0]])
    circuit.cx(q_c[4], q_c[0])
    circuit = circuit.compose(get_cat_disentangler_circuit(2), [q_d[0], q_d[4], q_c[4]], [c_d[0][0], c_d[4][0], c_c[4][0]])

    circuit.h(q_b[0])
    circuit.t(q_d[0])
    circuit.tdg(q_c[0])

    circuit = circuit.compose(get_cat_entangler_circuit(2), [q_d[0], q_d[4], q_c[4]], [c_d[0][0], c_d[4][0], c_c[4][0]])
    circuit.cx(q_c[4], q_c[0])
    circuit = circuit.compose(get_cat_disentangler_circuit(2), [q_d[0], q_d[4], q_c[4]], [c_d[0][0], c_d[4][0], c_c[4][0]])


    # FIRST NON LOCAL TOFFOLI END!!!

    circuit.h(q_a[0])
    circuit.tdg(q_b[0])
    circuit.tdg(q_a[0])

    circuit = circuit.compose(get_cat_entangler_circuit(2), [q_b[0], q_b[4], q_a[4]], [c_b[0][0], c_b[4][0], c_a[4][0]])
    circuit.cx(q_a[4], q_a[0])
    circuit = circuit.compose(get_cat_disentangler_circuit(2), [q_b[0], q_b[4], q_a[4]], [c_b[0][0], c_b[4][0], c_a[4][0]])

    circuit.t(q_a[0])

    circuit = circuit.compose(get_cat_entangler_circuit(2), [q_b[0], q_b[4], q_a[4]], [c_b[0][0], c_b[4][0], c_a[4][0]])
    circuit.cx(q_a[4], q_a[0])
    circuit = circuit.compose(get_cat_disentangler_circuit(2), [q_b[0], q_b[4], q_a[4]], [c_b[0][0], c_b[4][0], c_a[4][0]])

    circuit.h(q_a[0])

    ## SECOND NON LOCAL TOFFOLI
    circuit.h(q_b[0])

    circuit = circuit.compose(get_cat_entangler_circuit(2), [q_c[0], q_c[4], q_b[4]], [c_c[0][0], c_c[4][0], c_b[4][0]])
    circuit.cx(q_b[4], q_b[0])
    circuit = circuit.compose(get_cat_disentangler_circuit(2), [q_c[0], q_c[4], q_b[4]], [c_c[0][0], c_c[4][0], c_b[4][0]])

    circuit.tdg(q_b[0])

    circuit = circuit.compose(get_cat_entangler_circuit(2), [q_d[0], q_d[4], q_b[4]], [c_d[0][0], c_d[4][0], c_b[4][0]])
    circuit.cx(q_b[4], q_b[0])
    circuit = circuit.compose(get_cat_disentangler_circuit(2), [q_d[0], q_d[4], q_b[4]], [c_d[0][0], c_d[4][0], c_b[4][0]])

    circuit.t(q_b[0])

    circuit = circuit.compose(get_cat_entangler_circuit(2), [q_c[0], q_c[4], q_b[4]], [c_c[0][0], c_c[4][0], c_b[4][0]])
    circuit.cx(q_b[4], q_b[0])
    circuit = circuit.compose(get_cat_disentangler_circuit(2), [q_c[0], q_c[4], q_b[4]], [c_c[0][0], c_c[4][0], c_b[4][0]])

    circuit.tdg(q_b[0])

    circuit = circuit.compose(get_cat_entangler_circuit(2), [q_d[0], q_d[4], q_b[4]], [c_d[0][0], c_d[4][0], c_b[4][0]])
    circuit.cx(q_b[4], q_b[0])
    circuit = circuit.compose(get_cat_disentangler_circuit(2), [q_d[0], q_d[4], q_b[4]], [c_d[0][0], c_d[4][0], c_b[4][0]])

    circuit.t(q_b[0])
    circuit.t(q_c[0])

    circuit = circuit.compose(get_cat_entangler_circuit(2), [q_d[0], q_d[4], q_c[4]], [c_d[0][0], c_d[4][0], c_c[4][0]])
    circuit.cx(q_c[4], q_c[0])
    circuit = circuit.compose(get_cat_disentangler_circuit(2), [q_d[0], q_d[4], q_c[4]], [c_d[0][0], c_d[4][0], c_c[4][0]])

    circuit.h(q_b[0])
    circuit.t(q_d[0])
    circuit.tdg(q_c[0])

    circuit = circuit.compose(get_cat_entangler_circuit(2), [q_d[0], q_d[4], q_c[4]], [c_d[0][0], c_d[4][0], c_c[4][0]])
    circuit.cx(q_c[4], q_c[0])
    circuit = circuit.compose(get_cat_disentangler_circuit(2), [q_d[0], q_d[4], q_c[4]], [c_d[0][0], c_d[4][0], c_c[4][0]])
    ## SECOND NON LCOAL TOFFOLI END
    circuit.h(q_a[0])
    circuit.t(q_b[0])
    circuit.t(q_a[0])
    circuit = circuit.compose(get_cat_entangler_circuit(2), [q_b[0], q_b[4], q_a[4]], [c_b[0][0], c_b[4][0], c_a[4][0]])
    circuit.cx(q_a[4], q_a[0])
    circuit = circuit.compose(get_cat_disentangler_circuit(2), [q_b[0], q_b[4], q_a[4]], [c_b[0][0], c_b[4][0], c_a[4][0]])
    circuit.tdg(q_a[0])  # HERE I WAS LEFT CHECKING
    circuit = circuit.compose(get_cat_entangler_circuit(2), [q_b[0], q_b[4], q_a[4]], [c_b[0][0], c_b[4][0], c_a[4][0]])
    circuit.cx(q_a[4], q_a[0])
    circuit = circuit.compose(get_cat_disentangler_circuit(2), [q_b[0], q_b[4], q_a[4]], [c_b[0][0], c_b[4][0], c_a[4][0]])
    circuit.h(q_a[0])


    circuit = circuit.compose(get_cat_entangler_circuit(2), [q_d[0], q_d[4], q_c[4]], [c_d[0][0], c_d[4][0], c_c[4][0]])
    circuit.cx(q_c[4], q_c[0])
    circuit = circuit.compose(get_cat_disentangler_circuit(2), [q_d[0], q_d[4], q_c[4]], [c_d[0][0], c_d[4][0], c_c[4][0]])
    circuit.h(q_a[0])

    circuit.rz(-1/8 * np.pi, q_c[0])
    circuit.rz(-1/8 * np.pi, q_a[0])

    circuit = circuit.compose(get_cat_entangler_circuit(2), [q_c[0], q_c[4], q_a[4]], [c_c[0][0], c_c[4][0], c_a[4][0]])
    circuit.cx(q_a[4], q_a[0])
    circuit = circuit.compose(get_cat_disentangler_circuit(2), [q_c[0], q_c[4], q_a[4]], [c_c[0][0], c_c[4][0], c_a[4][0]])

    circuit.rz(1 / 8 * np.pi, q_a[0])

    circuit = circuit.compose(get_cat_entangler_circuit(2), [q_c[0], q_c[4], q_a[4]], [c_c[0][0], c_c[4][0], c_a[4][0]])
    circuit.cx(q_a[4], q_a[0])
    circuit = circuit.compose(get_cat_disentangler_circuit(2), [q_c[0], q_c[4], q_a[4]], [c_c[0][0], c_c[4][0], c_a[4][0]])

    circuit.h(q_a[0])

    circuit = circuit.compose(get_cat_entangler_circuit(2), [q_d[0], q_d[4], q_c[4]], [c_d[0][0], c_d[4][0], c_c[4][0]])
    circuit.cx(q_c[4], q_c[0])
    circuit = circuit.compose(get_cat_disentangler_circuit(2), [q_d[0], q_d[4], q_c[4]], [c_d[0][0], c_d[4][0], c_c[4][0]])
    circuit.h(q_a[0])

    circuit.rz(1/8 * np.pi, q_c[0])
    circuit.rz(1/8 * np.pi, q_a[0])

    circuit = circuit.compose(get_cat_entangler_circuit(2), [q_c[0], q_c[4], q_a[4]], [c_c[0][0], c_c[4][0], c_a[4][0]])
    circuit.cx(q_a[4], q_a[0])
    circuit = circuit.compose(get_cat_disentangler_circuit(2), [q_c[0], q_c[4], q_a[4]], [c_c[0][0], c_c[4][0], c_a[4][0]])

    circuit.rz(-1 / 8 * np.pi, q_a[0])

    circuit = circuit.compose(get_cat_entangler_circuit(2), [q_c[0], q_c[4], q_a[4]], [c_c[0][0], c_c[4][0], c_a[4][0]])
    circuit.cx(q_a[4], q_a[0])
    circuit = circuit.compose(get_cat_disentangler_circuit(2), [q_c[0], q_c[4], q_a[4]], [c_c[0][0], c_c[4][0], c_a[4][0]])
    circuit.h(q_a[0])
    circuit.h(q_a[0])
    circuit.rz(1/8 * np.pi, q_d[0])
    circuit.rz(1/8 * np.pi, q_a[0])

    circuit = circuit.compose(get_cat_entangler_circuit(2), [q_d[0], q_d[4], q_a[4]], [c_d[0][0], c_d[4][0], c_a[4][0]])
    circuit.cx(q_a[4], q_a[0])
    circuit = circuit.compose(get_cat_disentangler_circuit(2), [q_d[0], q_d[4], q_a[4]], [c_d[0][0], c_d[4][0], c_a[4][0]])

    circuit.rz(-1 / 8 * np.pi, q_a[0])

    circuit = circuit.compose(get_cat_entangler_circuit(2), [q_d[0], q_d[4], q_a[4]], [c_d[0][0], c_d[4][0], c_a[4][0]])
    circuit.cx(q_a[4], q_a[0])
    circuit = circuit.compose(get_cat_disentangler_circuit(2), [q_d[0], q_d[4], q_a[4]], [c_d[0][0], c_d[4][0], c_a[4][0]])

    circuit.h(q_a[0])
    ## NON LOCAL CCCX GATES --- END

    return circuit

if __name__ == '__main__':
    a = 0 #1 / sqrt(2)
    b = 1 #/ sqrt(2)
    circuit = get_shor_code_4_c_4('random', 'random', 'random', a, b)

    print(circuit.draw())
    print("Circuit depth: ",
          circuit.depth())  # measure at the end + error block (which might introduce extra gate) should be commented out
    circuit.measure(circuit.qregs[0][0], circuit.cregs[5*0 + 0])
    circuit.measure(circuit.qregs[0][1], circuit.cregs[5*0 + 1])
    circuit.measure(circuit.qregs[0][2], circuit.cregs[5*0 + 2])
    circuit.measure(circuit.qregs[0][3], circuit.cregs[5*0 + 3])
    circuit.measure(circuit.qregs[0][4], circuit.cregs[5*0 + 4])
    circuit.measure(circuit.qregs[1][0], circuit.cregs[5*1 + 0])
    circuit.measure(circuit.qregs[1][1], circuit.cregs[5*1 + 1])
    circuit.measure(circuit.qregs[1][2], circuit.cregs[5*1 + 2])
    circuit.measure(circuit.qregs[1][3], circuit.cregs[5*1 + 3])
    circuit.measure(circuit.qregs[1][4], circuit.cregs[5*1 + 4])
    circuit.measure(circuit.qregs[2][0], circuit.cregs[5*2 + 0])
    circuit.measure(circuit.qregs[2][1], circuit.cregs[5*2 + 1])
    circuit.measure(circuit.qregs[2][2], circuit.cregs[5*2 + 2])
    circuit.measure(circuit.qregs[2][3], circuit.cregs[5*2 + 3])
    circuit.measure(circuit.qregs[2][4], circuit.cregs[5*2 + 4])
    circuit.measure(circuit.qregs[3][0], circuit.cregs[5*3 + 0])
    circuit.measure(circuit.qregs[3][1], circuit.cregs[5*3 + 1])
    circuit.measure(circuit.qregs[3][2], circuit.cregs[5*3 + 2])
    circuit.measure(circuit.qregs[3][3], circuit.cregs[5*3 + 3])
    circuit.measure(circuit.qregs[3][4], circuit.cregs[5*3 + 4])

    QI_authenticate()
    qi_backend = QI.get_backend('QX single-node simulator')
    qi_job = execute(circuit, backend=qi_backend, shots=8)
    qi_result = qi_job.result()
    histogram = qi_result.get_counts(circuit)
    print('State\tCounts')
    [print('{0}\t{1}'.format(state, counts)) for state, counts in histogram.items()]

    # #Delete channel qubits from bit string to be printed
    # for state, counts in histogram.items():
    #     results_all = list(list(state))
    #     results_all = results_all[::2]
    #     results_all = "".join(results_all)
    #     results = []
    #     for i in range(len(results_all)):
    #         if i % 5 == 0:
    #             continue
    #         else:
    #             results.append(results_all[i])
    #     results = "".join(results)
    #     results = results + " " + str(counts)
    #     print(results)
