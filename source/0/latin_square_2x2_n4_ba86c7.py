# https://github.com/22slin22/quantum_algorithms/blob/59dd00f404c5283c8418b9e59e98f161f5d1bb04/latin_square_2x2_n4.py
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.visualization import plot_histogram

import matplotlib.pyplot as plt

simulator = Aer.get_backend('qasm_simulator')

row_constraints = [(c1 + row*2, c2 + row*2) for c1 in range(2) for c2 in range(2) for row in range(2) if c1 < c2]
column_constraints = [(row1*2 + column, row2*2 + column) for row1 in range(2) for row2 in range(2) for column in range(2) if row1 < row2]

clause_list = row_constraints + column_constraints
clause_list_len = len(clause_list)


def oracle(qc, clause_list, cells, ancillas1, ancillas2, output):
    for (c1, c2), a1, a2 in zip(clause_list, ancillas1, ancillas2):
        compare_cells(qc, cells[c1], cells[c2], a1, a2, reverse=False)

    # And all ancilla qubits together to check if latin square is valid
    qc.mcx(ancillas2, output)

    for (c1, c2), a1, a2 in zip(clause_list, ancillas1, ancillas2):
        compare_cells(qc, cells[c1], cells[c2], a1, a2, reverse=True)


def compare_cells(qc, cell1, cell2, ancilla1, ancilla2, reverse):
    if reverse:
        qc.x(ancilla2)
        qc.mcx(ancilla1, ancilla2)
        qc.x(ancilla1)

    # Compare first bit
    qc.cx(cell1[0], ancilla1[0])
    qc.cx(cell2[0], ancilla1[0])

    # Compare second bit
    qc.cx(cell1[1], ancilla1[1])
    qc.cx(cell2[1], ancilla1[1])

    if not reverse:
        qc.x(ancilla1)
        qc.mcx(ancilla1, ancilla2)
        qc.x(ancilla2)


def grover_diffusion(qc, cells):
    qc.h(all_qubits(cells))
    qc.x(all_qubits(cells))
    qc.h(cells[-1][-1])

    qc.mcx(all_qubits(cells)[:-1], cells[-1][-1])

    qc.h(cells[-1][-1])
    qc.x(all_qubits(cells))
    qc.h(all_qubits(cells))


def all_qubits(registers):
    return [q for r in registers for q in r]


cells = [QuantumRegister(2) for _ in range(4)]
ancillas1 = [QuantumRegister(2) for _ in range(clause_list_len)]
ancillas2 = QuantumRegister(clause_list_len)
output = QuantumRegister(1)
c = ClassicalRegister(8)

qc = QuantumCircuit(*cells, *ancillas1, ancillas2, output, c)


'''
Define algorithm
'''
qc.h(all_qubits(cells))

# Initialize output qubit to state |->
qc.x(output)
qc.h(output)

oracle(qc, clause_list, cells, ancillas1, ancillas2, output)
grover_diffusion(qc, cells)

qc.measure(all_qubits(cells), c)

# print(qc.draw())

print("Starting simulation")
job = execute(qc, simulator, shots=10000)

result = job.result()

print(result)

counts = result.get_counts(qc)
print("Counts are:", counts)

for k, v in counts.items():
    if v > 50:
        c1, c2, c3, c4 = k[0:2], k[2:4], k[4:6], k[6:8]
        c1, c2, c3, c4 = int(c1, base=2), int(c2, base=2), int(c3, base=2), int(c4, base=2)
        #print(c1, c2)
        #print(c3, c4)
        #print()
        print("\\latinTwoSmall{{{}}}{{{}}}{{{}}}{{{}}}".format(c1, c2, c3, c4))

print(sum(v for v in counts.values() if v > 50) / 10000)

plot_histogram(counts).show()
plt.show()
