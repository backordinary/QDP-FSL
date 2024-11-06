# https://github.com/22slin22/quantum_algorithms/blob/59dd00f404c5283c8418b9e59e98f161f5d1bb04/compare_3_cells.py
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.visualization import plot_histogram

import matplotlib.pyplot as plt

simulator = Aer.get_backend('qasm_simulator')


def compare_triplet(reverse):
    a, b, c = QuantumRegister(2), QuantumRegister(2), QuantumRegister(2)
    ancilla = QuantumRegister(5)
    output = QuantumRegister(1)

    qc = QuantumCircuit(a, b, c, ancilla, output)

    # Compare cells a and c
    qc.cx(a[0], ancilla[0])
    qc.cx(c[0], ancilla[0])

    qc.cx(a[1], ancilla[1])
    qc.cx(c[1], ancilla[1])

    qc.x(ancilla[0:2])

    qc.mcx(ancilla[0:2], ancilla[2])
    qc.x(ancilla[2])

    # Compare cells b and c
    qc.cx(b[0], c[0])
    qc.cx(b[1], c[1])

    qc.x(c)

    qc.mcx(c, ancilla[3])
    qc.x(ancilla[3])

    # Compare cells a and b
    qc.cx(a[0], b[0])
    qc.cx(a[1], b[1])

    qc.x(b)

    qc.mcx(b, ancilla[4])
    qc.x(ancilla[4])

    if not reverse:
        # Combine all comparisons
        qc.mcx(ancilla[2:5], output)

    if reverse:
        qc = qc.reverse_ops()

    print(qc.draw())

    compare = qc.to_gate()
    compare.name = "U$_cmp$"
    return compare


def grover_diffusion(qc, cells):
    qc.h(all_qubits(cells))
    qc.x(all_qubits(cells))
    qc.h(cells[-1][-1])

    qc.mcx(all_qubits(cells)[:-1], cells[-1][-1])

    qc.h(cells[-1][-1])
    qc.x(all_qubits(cells))
    qc.h(all_qubits(cells))


def all_qubits(registers):
    """
    Returns a list of qubits given a list of QuantumRegister
    :param registers: List of QuantumRegister
    :return: List of qubits
    """
    return [q for r in registers for q in r]


cells = [QuantumRegister(2) for _ in range(3)]
ancilla = QuantumRegister(5)
output = QuantumRegister(1)
c = ClassicalRegister(6)

qc = QuantumCircuit(*cells, ancilla, output, c)


'''
Define algorithm
'''
qc.h(all_qubits(cells))

# Initialize output qubit to state |->
qc.x(output)
qc.h(output)

qc.append(compare_triplet(reverse=False), list(range(12)))
qc.append(compare_triplet(reverse=True), list(range(12)))
grover_diffusion(qc, cells)

qc.measure(all_qubits(cells), c)

print(qc.draw())

print("Starting simulation")
job = execute(qc, simulator, shots=10000)

result = job.result()

print(result)

counts = result.get_counts(qc)
print("Counts are:", counts)

for k, v in counts.items():
    if v > 200:
        c1, c2, c3 = k[0:2], k[2:4], k[4:6]
        c1, c2, c3 = int(c1, base=2), int(c2, base=2), int(c3, base=2)
        print(c1, c2, c3)

plot_histogram(counts).show()
plt.show()
