# https://github.com/parasol4791/quantumComp/blob/5e20695c5800fd0024145f0b04d25b2fc8b3bd1c/sandbox.py
from qiskit import QuantumCircuit, assemble, Aer, execute
from qiskit.circuit.library import Diagonal
from qiskit.visualization import array_to_latex, plot_bloch_multivector
from pylatex import Document
from math import sqrt, pi
from gates import *


def process(qc, mode):
    if mode == 's':
        qc.save_statevector()
    elif mode == 'u':
        qc.save_unitary()

    result = execute(qc, sim).result()
    if mode == 's':
        s = result.get_statevector()
        print(s)
        print(result.data())
        sl = array_to_latex(s)
        print(sl.data)
        plot_bloch_multivector(s)
    elif mode == 'u':
        u = result.get_unitary()
        # print(u)
        ul = array_to_latex(u)
        print(ul.data)
    elif mode == 'b':
        plot_bloch_multivector(qc)


sp = 1/sqrt(2)
sim = Aer.get_backend('aer_simulator')


mode = 'u'  # s - state, u - unitary, b - bloch
n = 2
all_qubits = [q for q in range(n)]

#qc = QuantumCircuit(n)
from qiskit_textbook.problems import grover_problem_oracle
qc = grover_problem_oracle(n, 0)
qc = Diagonal([-1,1,1,1])
print(f"n={n}, w={0}")
print(qc.decompose().decompose().decompose().draw())
process(qc, mode)
all_qubits = [q for q in range(n)]

#qc = QuantumCircuit(n)
from qiskit_textbook.problems import grover_problem_oracle
qc = grover_problem_oracle(n, 1)
qc = Diagonal([-1,1,1,1])
print(f"n={n}, w={1}")
print(qc.decompose().decompose().decompose().draw())
process(qc, mode)
all_qubits = [q for q in range(n)]

#qc = QuantumCircuit(n)
from qiskit_textbook.problems import grover_problem_oracle
qc = grover_problem_oracle(n, 2)
qc = Diagonal([-1,1,1,1])
print(f"n={n}, w={2}")
print(qc.decompose().decompose().decompose().draw())
process(qc, mode)
all_qubits = [q for q in range(n)]

#qc = QuantumCircuit(n)
from qiskit_textbook.problems import grover_problem_oracle
qc = grover_problem_oracle(n, 3)
qc = Diagonal([-1,1,1,1])
print(f"n={n}, w={3}")
print(qc.decompose().decompose().decompose().draw())
process(qc, mode)










