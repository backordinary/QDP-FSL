# https://github.com/Schwarf/qiskit_fundamentals/blob/ba66945b4b57055cbd207e32c4d7105b8b1b01f3/quantum_circuits/bernstein_vazirani_algorithm.py
import numpy as np
from matplotlib import pyplot as plt
from qiskit import Aer, IBMQ
from qiskit import QuantumCircuit, assemble, transpile
from qiskit.circuit import Gate
from qiskit.providers.ibmq import least_busy
from qiskit.visualization import plot_histogram
from qiskit.tools.monitor import job_monitor

def bernstein_vazirani_circuit(oracle_gate: Gate, number_of_inputs: int) -> QuantumCircuit:
    number_of_bits = number_of_inputs
    number_of_qubits = number_of_inputs + 1
    bv_circuit = QuantumCircuit(number_of_qubits, number_of_bits)
    index_last_qubit = number_of_inputs
    # all qubits initialize to |0> ... last one shall be initialized to |-> ...0-indexed: ZH|0> = |->
    bv_circuit.h(index_last_qubit)
    bv_circuit.z(index_last_qubit)
    # Apply Hadamard to all input-qubits and the last one
    for qubit in range(number_of_inputs):
        bv_circuit.h(qubit)
    # Apply oracle function: Note this function is applied to all qubits, although is has only an effect on the last
    # qubit
    bv_circuit.append(oracle_gate, range(number_of_qubits))
    # Apply Hadamard to input qubits, the last qubit can now be ignored (in the next step measurement only applied to
    # input)
    for qubit in range(number_of_inputs):
        bv_circuit.h(qubit)

    for classic_bit in range(number_of_bits):
        bv_circuit.measure(classic_bit, classic_bit)

    return bv_circuit


def construct_oracle(hidden_string: str) -> Gate:
    number_of_inputs = len(hidden_string)
    number_of_outputs = 1
    number_of_qubits = number_of_inputs + number_of_outputs
    # revert string since qiskit is little-endian
    hidden_string = hidden_string[::-1]
    oracle_qc = QuantumCircuit(number_of_qubits)
    for qubit in range(number_of_inputs):
        if hidden_string[qubit] == '0':
            oracle_qc.i(qubit)
        else:
            oracle_qc.cx(qubit, number_of_inputs)

    oracle_gate = oracle_qc.to_gate()
    oracle_gate.name = "Bernstein-Vazirani-Oracle"
    return oracle_gate


number_of_inputs = 4
random_binary_string_number = np.random.randint(1, 2 ** number_of_inputs)
random_binary_string = format(random_binary_string_number, '0' + str(number_of_inputs) + 'b')
print(random_binary_string)
#random_binary_string = "11101101"

oracle_gate = construct_oracle(hidden_string=random_binary_string)
bv_circuit = bernstein_vazirani_circuit(oracle_gate=oracle_gate, number_of_inputs=number_of_inputs)

aer_sim = Aer.get_backend('aer_simulator')
transpiled_bv_circuit = transpile(bv_circuit, aer_sim)
"""
shots = 1024
qobj = assemble(transpiled_bv_circuit)
results = aer_sim.run(qobj).result()
answer = results.get_counts()

plot_histogram(answer)

plt.show()

"""
################################
# Run on a realistic device
################################

IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q')
provider.backends()
backend = least_busy(provider.backends(filters=lambda x: x.configuration().n_qubits <6 and x.configuration().n_qubits > 1 and
                                       not x.configuration().simulator and x.status().operational==True))

print("least busy backend: ", backend)
shots = 1024
transpiled_bv_circuit = transpile(transpiled_bv_circuit, backend)
job = backend.run(transpiled_bv_circuit, shots=shots)

job_monitor(job, interval=2)
results = job.result()
answer = results.get_counts()
plot_histogram(answer)

plt.show()