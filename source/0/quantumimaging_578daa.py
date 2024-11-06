# https://github.com/sergey55568/quantum/blob/2ff7d8bec68a9286f8ca593f6581f7e0834f7c0f/src/quantum_imaging_project/QuantumImaging.py
import matplotlib.pyplot as plt
from qiskit import (
    QuantumCircuit,
    execute,
    Aer)
from qiskit.providers.aer.noise.errors.standard_errors import pauli_error
from qiskit.quantum_info import Operator, Kraus

simulator = Aer.get_backend('qasm_simulator')

circuit = QuantumCircuit(12, 4)

p_error = 0.3
bit_flip = pauli_error([('X', p_error), ('I', 1 - p_error)])
idn = Kraus(bit_flip)

p_error_f = 0.15
bit_flip_f = pauli_error([('X', p_error_f), ('I', 1 - p_error_f)])
idnf = Kraus(bit_flip_f)


def safe_id(i):
    for j in [4, 5, 6, 7, 8, 9, 10, 11]:
        circuit.append(idnf, [j])

    circuit.cx(0, 4)
    circuit.cx(0, 5)
    circuit.h(0)
    circuit.h(4)
    circuit.h(5)
    circuit.cx(0, 6)
    circuit.cx(0, 7)
    circuit.cx(4, 8)
    circuit.cx(4, 9)
    circuit.cx(5, 10)
    circuit.cx(5, 11)

    # circuit.id(i)
    circuit.append(idn, [i])

    circuit.cx(0, 6)
    circuit.cx(0, 7)
    circuit.cx(4, 8)
    circuit.cx(4, 9)
    circuit.cx(5, 10)
    circuit.cx(5, 11)

    circuit.ccx(6, 7, 0)
    circuit.ccx(8, 9, 4)
    circuit.ccx(10, 11, 5)

    circuit.h(0)
    circuit.h(4)
    circuit.h(5)
    circuit.cx(0, 4)
    circuit.cx(0, 5)
    circuit.ccx(4, 5, 0)


def Q(t, s):
    T = Operator([[t, pow((1 - pow(t, 2)), 1 / 2)], [pow((1 - pow(t, 2)), 1 / 2), t]]).to_instruction()
    T.label = "T"
    circuit.cx(0, 1)
    circuit.id(2)
    circuit.id(3)

    if s:
        safe_id(0)
    else:
        circuit.append(idn, [0])

    circuit.append(T, [1])
    circuit.id(2)
    circuit.id(3)
    circuit.id(0)
    circuit.cx(1, 2)
    circuit.id(3)

    # Map the quantum measurement to the classical bits
    circuit.measure([0], [0])


Q(0, True)

job = execute(circuit, simulator, shots=2048, optimization_level=0)

result = job.result()

counts = result.get_counts(circuit)
print("\nTotal count for 0110 and 0111 are:", counts)

circuit.draw(output='mpl')
plt.show()
