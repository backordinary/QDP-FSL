# https://github.com/raz-mon/Quantum-Computation-Project/blob/b4ddeb791459b60cc9cfe3e1fc2b7cffa1c88530/util_real.py
import numpy as np
from qiskit.extensions import UnitaryGate
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import IBMQ, transpile
from qiskit.visualization import plot_histogram
import whole_process

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, Aer, execute
from qiskit.quantum_info import random_statevector, state_fidelity, Statevector
from qiskit.extensions import Initialize, UnitaryGate


# Here are 3 options of scrambling operators:
# This is the exact matrix from eq.4 in Methods from the main article (scrambling), multiplied by i (j in python).
mat1 = 0.5j * np.array([[-1, 0, 0, -1, 0, -1, -1, 0], [0, 1, -1, 0, -1, 0, 0, 1], [0, -1, 1, 0, -1, 0, 0, 1],
                        [1, 0, 0, 1, 0, -1, -1, 0], [0, -1, -1, 0, 1, 0, 0, 1], [1, 0, 0, -1, 0, 1, -1, 0],
                        [1, 0, 0, -1, 0, -1, 1, 0], [0, -1, -1, 0, -1, 0, 0, -1]])

# As mat1, but with a (-) factor.
mat2 = -0.5j * np.array([[-1, 0, 0, -1, 0, -1, -1, 0], [0, 1, -1, 0, -1, 0, 0, 1], [0, -1, 1, 0, -1, 0, 0, 1],
                         [1, 0, 0, 1, 0, -1, -1, 0], [0, -1, -1, 0, 1, 0, 0, 1], [1, 0, 0, -1, 0, 1, -1, 0],
                         [1, 0, 0, -1, 0, -1, 1, 0], [0, -1, -1, 0, -1, 0, 0, -1]])

# As mat1, with a (-j) factor).
mat3 = 0.5 * np.array([[-1, 0, 0, -1, 0, -1, -1, 0], [0, 1, -1, 0, -1, 0, 0, 1], [0, -1, 1, 0, -1, 0, 0, 1],
                       [1, 0, 0, 1, 0, -1, -1, 0], [0, -1, -1, 0, 1, 0, 0, 1], [1, 0, 0, -1, 0, 1, -1, 0],
                       [1, 0, 0, -1, 0, -1, 1, 0], [0, -1, -1, 0, -1, 0, 0, -1]])

scramble_gate2_1 = UnitaryGate(mat1)
scramble_gate2_1.label = 'scrambler_2_1'
scramble_gate2_1_c = UnitaryGate(mat1.conjugate())
scramble_gate2_1_c.label = 'scrambler_2_1_c'

scramble_gate2_2 = UnitaryGate(mat1)
scramble_gate2_2.label = 'scrambler_2_2'
scramble_gate2_2_c = UnitaryGate(mat2.conjugate())
scramble_gate2_2_c.label = 'scrambler_2_2_c'

scramble_gate2_3 = UnitaryGate(mat3)
scramble_gate2_3.label = 'scrambler_2_3'
scramble_gate2_3_c = UnitaryGate(mat3.conjugate())
scramble_gate2_3_c.label = 'scrambler_2_3_c'


# Make a circuit that makes a bell pair out of the two given qubits, give it a name.
def create_bell_pair(qc, a, b):
    """Creates a bell pair in QuantumCircuit qc between qubits a and b"""
    qc.h(a)  # Put qubit a into state |+>
    qc.cx(a, b)  # CNOT with a as control and b as target


qreg = QuantumRegister(2, 'q')
circ = QuantumCircuit(qreg)
create_bell_pair(circ, qreg[0], qreg[1])

bell_gate = circ.to_gate(label='bell')

inv_Bell_circ = circ.inverse()
inv_Bell_gate = inv_Bell_circ.to_gate(label='inv_bell')


def circ25_real():
    """
    Exactly as circ25, only here we eliminate the measurements of all qubits. This is done so the system
    quantum state won't collapse, which is necessary for the measurement of the state fidelity
    """

    # circuit will be our main circuit.
    qreg_q = QuantumRegister(7, 'q')
    creg_c = ClassicalRegister(7, 'c')
    circuit = QuantumCircuit(qreg_q, creg_c)

    # q0 <- px0. Initialize the first qubit with the first eigenstate of pauli x (corresponding to eigenvalue 1 - |+>).
    circuit.h(qreg_q[0])

    # Make Bell pairs from the qubits (0, 3), (1, 4), (2, 5), which corresponds to the tfd state of
    # maximum correlation (infinite temperature).
    # create_bell_pair(circuit, qreg_q[0], qreg_q[3])
    # create_bell_pair(circuit, qreg_q[1], qreg_q[4])
    # create_bell_pair(circuit, qreg_q[2], qreg_q[5])
    # Here we initialize qubits 1-6 states as the tfd, which we constructed in Mathematica:
    psi2 = whole_process.tfd_generator().generate_tfd(0.25)
    init_gate_1_6 = Initialize(psi2)
    init_gate_1_6.label = 'init_gate 1-6'

    # Add the initializing gate to qubits 1-6.
    circuit.append(init_gate_1_6, [qreg_q[1], qreg_q[2], qreg_q[3], qreg_q[4], qreg_q[5], qreg_q[6]])

    circuit.barrier()

    # Apply the scrambling unitary to qubits 0-2 and U^* to 3-5.
    circuit.append(scramble_gate2_1, [0, 1, 2])
    circuit.append(scramble_gate2_1_c, [5, 4, 3])

    circuit.barrier()

    circuit.append(inv_Bell_gate, [qreg_q[0], qreg_q[3]])
    circuit.append(inv_Bell_gate, [qreg_q[1], qreg_q[4]])
    circuit.append(inv_Bell_gate, [qreg_q[2], qreg_q[5]])

    circuit.barrier()

    # Append to the circuit the inverse of 'init_gate' on qubit 6 (the 'target'):
    # circuit.append(inverse_init_gate, [qreg_q[6]])
    circuit.h(qreg_q[6])            # Append the inverse gate of the initializing gate of q0.

    # Measure qubits 2, 5. Save result in corresponding classical bits.
    circuit.measure([qreg_q[2], qreg_q[5]], [creg_c[2], creg_c[5]])
    # Add the measurement of qubit 6, to classical bit 6:
    circuit.measure(qreg_q[6], creg_c[6])

    IBMQ.load_account()
    provider = IBMQ.get_provider(hub='ibm-q-research-2', group='ben-gurion-uni-1', project='main')

    # get the least-busy backend at IBM and run the quantum circuit there
    from qiskit.providers.ibmq import least_busy
    from qiskit.tools.monitor import job_monitor

    backend = least_busy(provider.backends(filters=lambda b: b.configuration().n_qubits >= 7 and
                                                             not b.configuration().simulator and b.status().operational == True))
    t_qc = transpile(circuit, backend, optimization_level=1)
    job = backend.run(t_qc)
    job_monitor(job)  # displays job status under cell

    # Get the results and display them
    exp_result = job.result()
    exp_counts = exp_result.get_counts(circuit)
    good = exp_counts.get('0000000', 0)
    bad = exp_counts.get('1000000', 0)
    # Fidelity - TBD.
    print('successful measurement probability: ', good/1024)
    print('unsuccessful teleportation counts: ', bad/1024)
    # print(exp_counts)
    plot_histogram(exp_counts)


circ25_real()
