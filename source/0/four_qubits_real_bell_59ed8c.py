# https://github.com/Gitiauxx/QuantDist/blob/ec9c7f6030bcd027b662822d0df845ea7c8f01cc/examples/four_qubits_real_bell.py
import os

import numpy as np
from  math import sqrt
import matplotlib.pyplot as plt

from qiskit import IBMQ
#IBMQ.save_account('d5c3abdfb2d464260eeda4fc0aecd1ed1e3e64c270703dd5a0a34a1546f57c7d7a1e0d19d695c0adc43f509b7219cff4f86da11e818edcf095dfab7403649c1f')
IBMQ.load_account()

from qiskit import QuantumCircuit, QuantumRegister, execute, Aer, ClassicalRegister, BasicAer
from qiskit.compiler import transpile, assemble
from qiskit.visualization import plot_gate_map
from qiskit.test.mock import FakeMelbourne

from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import pauli_error, depolarizing_error

from source.utils import get_logger

logger = get_logger(__name__)

def get_noise(p_meas, p_gate):
    error_meas = pauli_error([('X', p_meas), ('I', 1 - p_meas)])
    error_gate1 = depolarizing_error(p_gate, 1)
    error_gate2 = error_gate1.tensor(error_gate1)

    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_meas, "measure")
    noise_model.add_all_qubit_quantum_error(error_gate1, ["x"])
    noise_model.add_all_qubit_quantum_error(error_gate2, ["cx"])

    return noise_model


def construct_circuit_full(plot=False):
    r0 = [1, 0, 0, 0]
    r1 = [sqrt(5 / 9), sqrt(4 / 9), 0, 0]
    r2 = [sqrt(5 / 9), -sqrt(1 / 9), 0. + 1j * sqrt(1 / 3), 0]
    r3 = [sqrt(5 / 9), -sqrt(1 / 9), 0. - 1j * sqrt(1 / 3), 0]

    initial_state_list = [r0, r1, r2, r3]

    qr = QuantumRegister(8, 'q')
    anc = QuantumRegister(2, 'ancilla')
    cr = ClassicalRegister(2, 'c_anc')
    cr_q = ClassicalRegister(8, 'c_q')
    circuit = QuantumCircuit(anc, qr, cr, cr_q)

    for i, r in enumerate(initial_state_list):
        circuit.initialize(r, [qr[2 * i], qr[2 * i + 1]])

    circuit.h(anc[0])
    circuit.h(anc[1])

    # swap btw qr[0] and qr[2]
    circuit.cnot(qr[2], qr[0])

    circuit.h(qr[2])
    circuit.cnot(qr[0], qr[2])
    circuit.tdg(qr[2])
    circuit.cnot(anc[1], qr[2])
    circuit.t(qr[2])
    circuit.cnot(qr[0], qr[2])
    circuit.tdg(qr[2])
    circuit.cnot(anc[1], qr[2])
    circuit.t(qr[0])
    circuit.t(qr[2])
    circuit.cnot(anc[1], qr[0])
    circuit.h(qr[2])
    circuit.t(anc[0])
    circuit.tdg(qr[0])
    circuit.cnot(anc[1], qr[0])

    circuit.cnot(qr[2], qr[0])

    # swap btw qr[1] and qr[3]
    circuit.cnot(qr[3], qr[1])

    circuit.h(qr[3])
    circuit.cnot(qr[1], qr[3])
    circuit.tdg(qr[3])
    circuit.cnot(anc[1], qr[3])
    circuit.t(qr[3])
    circuit.cnot(qr[1], qr[3])
    circuit.tdg(qr[3])
    circuit.cnot(anc[1], qr[3])
    circuit.t(qr[1])
    circuit.t(qr[3])
    circuit.cnot(anc[1], qr[1])
    circuit.h(qr[3])
    circuit.t(anc[1])
    circuit.tdg(qr[1])
    circuit.cnot(anc[1], qr[1])

    circuit.cnot(qr[3], qr[1])

    # swap btw qr[2] and qr[4]
    circuit.cnot(qr[4], qr[2])

    circuit.h(qr[4])
    circuit.cnot(qr[2], qr[4])
    circuit.tdg(qr[4])
    circuit.cnot(anc[0], qr[4])
    circuit.t(qr[4])
    circuit.cnot(qr[2], qr[4])
    circuit.tdg(qr[4])
    circuit.cnot(anc[0], qr[4])
    circuit.t(qr[2])
    circuit.t(qr[4])
    circuit.cnot(anc[0], qr[2])
    circuit.h(qr[4])
    circuit.t(anc[0])
    circuit.tdg(qr[2])
    circuit.cnot(anc[0], qr[2])

    circuit.cnot(qr[4], qr[2])

    # swap qr[3] and qr[5]
    circuit.cnot(qr[5], qr[3])

    circuit.h(qr[5])
    circuit.cnot(qr[3], qr[5])
    circuit.tdg(qr[5])
    circuit.cnot(anc[0], qr[5])
    circuit.t(qr[5])
    circuit.cnot(qr[3], qr[5])
    circuit.tdg(qr[5])
    circuit.cnot(anc[0], qr[5])
    circuit.t(qr[3])
    circuit.t(qr[5])
    circuit.cnot(anc[0], qr[3])
    circuit.h(qr[5])
    circuit.t(anc[0])
    circuit.tdg(qr[3])
    circuit.cnot(anc[0], qr[3])

    circuit.cnot(qr[5], qr[3])

    circuit.cnot(qr[0], qr[2])
    circuit.cnot(qr[1], qr[3])

    circuit.cnot(qr[4], qr[6])
    circuit.cnot(qr[5], qr[7])

    circuit.h(qr[0])
    circuit.h(qr[1])
    circuit.h(qr[4])
    circuit.h(qr[5])

    circuit.measure(qr, cr_q)
    circuit.measure(anc, cr)

    if plot:
        circuit.draw(output='mpl')
        plt.show()

    return circuit, qr, anc

def bitwise_and(x, y):

    assert len(x) == len(y)

    total = 0
    for i in range(len(x)):
        total += int(x[i]) * int(y[i])

    return total

def run_job(circuit, backend, layout=None, real_machine=True, noise=None):

    if real_machine:
        mapped_circuit = transpile(circuit, backend=backend, optimization_level=3, initial_layout=layout)
    else:
        mapped_circuit = transpile(circuit, backend=backend)

    logger.info(f'Depth: {mapped_circuit.depth()}')
    logger.info(f'Gates count: {mapped_circuit.count_ops()}')

    qobj = assemble(mapped_circuit, backend=backend, shots=8192, noise_model=noise)

    job = backend.run(qobj)
    result = job.result()

    return result, mapped_circuit

def count(result):
    counts = result.get_counts(circuit)
    total = sum(counts.values())

    counts = {state: c / total for state, c in counts.items()}

    r12_state = 0
    for state, c in counts.items():
        if (state[-2:] == '00') & (bitwise_and(state[4:6], state[6:8]) % 2 == 0):
            r12_state += c
    r12 = r12_state * 2 ** 3 - 1

    r13_state = 0
    for state, c in counts.items():
        if (state[-2:] == '01') & (bitwise_and(state[4:6], state[6:8]) % 2 == 0):
            r13_state += c
    r13 = r13_state * 2 ** 3 - 1

    r14_state = 0
    for state, c in counts.items():
        if (state[-2:] == '11') & (bitwise_and(state[:2], state[2:4]) % 2 == 0):
            r14_state += c
    r14 = r14_state * 2 ** 3 - 1

    r23_state = 0
    for state, c in counts.items():
        if (state[-2:] == '11') & (bitwise_and(state[4:6], state[6:8]) % 2 == 0):
            r23_state += c
    r23 = r23_state * 2 ** 3 - 1

    r24_state = 0
    for state, c in counts.items():
        if (state[-2:] == '10') & (bitwise_and(state[:2], state[2:4]) % 2 == 0):
            r24_state += c
    r24 = r24_state * 2 ** 3 - 1

    r34_state = 0
    for state, c in counts.items():
        if (state[-2:] == '00') & (bitwise_and(state[:2], state[2:4]) % 2 == 0):
            r34_state += c
    r34 = r34_state * 2 ** 3 - 1

    logger.info(f'LHS of face is {r12 + r13 + r14 - r23 - r24 - r34}')

    return r12 + r13 + r14 - r23 - r24 - r34


circuit, qr, anc = construct_circuit_full()

layout = {anc[1]: 0, qr[2]: 2, qr[1]: 14, qr[3]: 13, qr[0]: 1,
          qr[5]: 9, qr[4]: 3, anc[0]: 8, qr[6]: 4, qr[7]: 10}

provider = IBMQ.get_provider(hub='ibm-q')
backend = Aer.get_backend('qasm_simulator') #provider.get_backend('ibmq_16_melbourne')
result, mapped_circuit = run_job(circuit, backend, real_machine=False)
# simulator =  Aer.get_backend('qasm_simulator')
# job = execute(circuit, simulator, shots=20000)
# result = job.result()


iterations = 10
n_p_gates =  10

p_meas = 0.01
results = np.empty((iterations, n_p_gates, 4))

p_gate = np.linspace(0, 0.02, n_p_gates)

for i in range(iterations):

    for j in range(n_p_gates):

        noise = get_noise(p_meas, p_gate[j])
        logger.info(f'Iteration {i}')
        result, mapped_circuit = run_job(circuit, backend, real_machine=False, noise=noise)
        lhs = count(result)

        results[i, j, 0] = mapped_circuit.depth()

        gate_counts = mapped_circuit.count_ops()
        results[i, j, 1] = gate_counts['cx']
        results[i, j, 2] = lhs
        results[i, j, 3] = p_gate[j]

# accuracy = (results[:, 2] >= 1).astype('int32').mean()
# logger.info(f'Accuracy of quantum-classic classifier is {accuracy}')

results_folder = '../results/six_qubit'
os.makedirs(results_folder, exist_ok=True)
np.save(f'{results_folder}/max_noncoherent_six_with_noise_meas_error_{p_meas}.npy', results)

