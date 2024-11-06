# https://github.com/StijnW66/Quantum-Project/blob/f9b6abb906e89f70e665dd53b408e07fabab4b7d/src/quantum/gates/controlled_multiplier_gate.py
import sys
sys.path.append(".")

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import QFT, MCMT

from src.quantum.gates.modular_adder_gate import modular_adder


def controlled_multiplier_gate(x_size, a, N):
    b_size = len(bin(N).lstrip("0b"))
    # Although one additional controlling qubit 'c1' is specified for controlled
    # multiplier gate, the 'c2' qubit is to match qubit control of modular adder
    c1 = QuantumRegister(1, 'c')
    x = QuantumRegister(x_size, 'x')
    b = QuantumRegister(b_size, 'b')
    c2 = QuantumRegister(1, 'mod_add_0')

    cr = QuantumCircuit(c1, x, b, c2)

    # Create QFT gate and apply to 'b' qubits
    qft_b = QFT(num_qubits=b_size, approximation_degree=0, do_swaps=True, inverse=False, insert_barriers=False,
                name='qft')
    cr.append(qft_b.to_instruction(), b)

    # Create Modular Adder Gates, set control to c1 and x[i] qubits and add to circuit
    for i in range(x_size):
        temp_a = (2 ** i) * a
        temp_modular_adder = modular_adder(temp_a, N, b_size)
        temp_modular_adder.name = "mod_add_" + str(2) + "^" + str(i) + "a_N"
        cr.append(temp_modular_adder, [c1[0], x[i]] + b[0:b_size] + [c2[0]])

    # Create QFT^(-1) gate and apply to 'b' qubits
    inv_qft_b = QFT(num_qubits=b_size, approximation_degree=0, do_swaps=True, inverse=True, insert_barriers=False,
                    name='iqft')

    cr.append(inv_qft_b.to_instruction(), b)

    return cr

def new_controlled_multiplier_gate(size, a, N):
    c1 = QuantumRegister(1)
    x = QuantumRegister(size)
    b = QuantumRegister(size + 1) # add + 1 to account for overflow.
    c2 = QuantumRegister(1)

    circuit = QuantumCircuit(c1, x, b, c2)

    circuit.append(QFT(num_qubits=size + 1, approximation_degree=0, do_swaps=True, inverse=False, insert_barriers=False, name='qft'), b)
    for i in range(size):
        circuit.append(modular_adder((2**i * a) % N, N, size + 1), [c1[0], x[i]] + b[0:size+1] + [c2[0]])
    circuit.append(QFT(num_qubits=size + 1, approximation_degree=0, do_swaps=True, inverse=True, insert_barriers=False, name='iqft'), b)
    return circuit
