# https://github.com/lbadams2/csc_qc/blob/c9894fa587a331e44583717341013abd3a5d6875/hw3/parallel_xor.py
'''
Liam Adams - lbadams2
'''
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.quantum_info.operators import Operator
from qiskit import Aer, execute
from qiskit.tools.visualization import plot_histogram
import matplotlib.pyplot as plt


def create_xor_operator(n):
    num_rows = 2**n
    num_cols = num_rows
    unitary = []
    for i in range(num_rows):
        if i < num_rows - 2:
            row = [0 if j != i else 1 for j in range(num_cols)]
        elif i == num_rows - 2:
            row = [0 if j != i + 1 else 1 for j in range(num_cols)]
        else:
            row = [0 if j != i - 1 else 1 for j in range(num_cols)]
        unitary.append(row)
    
    #for r in unitary:
    #    print(r)
    u = Operator(unitary)
    return u


def create_circuit(n):
    n += 1 # need top ancilla qubit equal to 1
    u = create_xor_operator(n)
    a = QuantumRegister(n, 'in')  # input
    qc = QuantumCircuit(a)
    qc.x(a[0]) # make top qubit 1
    for i in range(1, n):
        qc.h(a[i])
    q_list = [i for i in range(n)]
    qc.unitary(u, q_list, label='U')

    m = ClassicalRegister(1, 'output')
    qc.add_register(m)
    qc.measure(a[n-1], m)

    return qc


if __name__ == '__main__':
    qc = create_circuit(6) # using 13 here will make your laptop crash

    backend = Aer.get_backend('qasm_simulator')
    job = execute(qc, backend, shots=4096)  # shots default = 1024
    result = job.result()
    print(result.get_counts())
    qc.draw(output='mpl')
    plot_histogram(result.get_counts())
    plt.show()