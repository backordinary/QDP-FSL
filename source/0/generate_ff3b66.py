# https://github.com/mtweiden/circuit_generators/blob/b160e052e651209dc70d74465b930ad7b333eb12/generators/qft/generate.py
from qiskit.circuit.library.basis_change.qft import QFT
from qiskit import QuantumCircuit
from qiskit.compiler.transpiler import transpile
from bqskit import Circuit
import pickle

if __name__ == '__main__':
    range_low = 3
    range_high = 65
    num_qubits = [x for x in range(range_low, range_high+1)]

    qasm_list = []

    for i, num_q in enumerate(num_qubits):
        qiskit_circ : QuantumCircuit = QFT(num_q)
        qiskit_circ = transpile(qiskit_circ, basis_gates=['u3','cx'])

        with open(f'temp_qasm/{i}.qasm', 'w') as f:
            f.write(qiskit_circ.qasm())
        circuit : Circuit = Circuit.from_file(f'temp_qasm/{i}.qasm')
        qasm_list.append(circuit)

    with open(f'qft_list_{range_low}_{range_high}.pickle', 'wb') as f:
        pickle.dump(qasm_list, f)