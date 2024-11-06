# https://github.com/FelixiaV/Deutsch-Jozsa-in-FakeVigo/blob/016f9d875ae2880a31d92060aaeb883dd67de376/deutsch-jozsa.py
"""
    Deutsch-Jozsa example in FakeVigo
    The Qiskit Textbook resource was used.
"""

# Qiskit importation
import numpy as np
import matplotlib.pyplot as plt

# Qiskit importation
from qiskit import Aer
from qiskit import QuantumCircuit, assemble, transpile
from qiskit.visualization import plot_histogram
from qiskit.test.mock import FakeVigo

#######################################################################
#                        Making Oracle
#######################################################################

def Oracle_Balanced(n):
    """
        It takes condition as string and number of qubits as integer.
        Oracle with n+1 qubits.
    """
    oracle_qc = QuantumCircuit(n+1)
    
    a = [0,1]
    b_str = "" # String with 0's and 1's. X gate will be applied to 1's.
    for _ in range(n):
        b_str += str(a[np.random.randint(0,1)])

    for qubit in range(len(b_str)):
        if b_str[qubit] == '1':
            oracle_qc.x(qubit)

    for qubit in range(n):
        oracle_qc.cx(qubit, n)
        if b_str[qubit] == '1':
            oracle_qc.x(qubit)

    return oracle_qc.to_gate()

def Oracle_Constant(n):
    """
        It takes number of qubits as integer.
        It returns oracle gate in constant gate.
    """
    oracle_qc = QuantumCircuit(n+1)

    if np.random.randint(2) == 1: # Output of the oracle will be decied (0 or 1)
        oracle_qc.x(n)
    
    return oracle_qc.to_gate()

#######################################################################
#             Making Circuit for Deutsch Jozsa ALgorithm
######################################################################

def Deutsch_Jozsa_Algorithm(oracle, n):
    Deutsch_Jozsa_Circuit = QuantumCircuit(n+1, n)

    Deutsch_Jozsa_Circuit.x(n)
    Deutsch_Jozsa_Circuit.h(n)


    Deutsch_Jozsa_Circuit.h(np.arange(n))

    Deutsch_Jozsa_Circuit.append(oracle, range(n+1))

    Deutsch_Jozsa_Circuit.h(np.arange(n))

    Deutsch_Jozsa_Circuit.measure(np.arange(n), np.arange(n))
    
    return Deutsch_Jozsa_Circuit

#######################################################################
#                           Main
#######################################################################
def main():
    n = int(input("Please write the qubit number you want: "))
    condition = input("Please write the condition you want to deal in oracle. (consant or balanced): ")
    Fake_Vigo_Sim = FakeVigo()

    if condition.lower() == "balanced":
        oracle_gate = Oracle_Balanced(n)
    elif condition.lowe == "constant":
        oracle_gate = Oracle_Constant(n)

    Deutsch_Jozsa_Circuit = Deutsch_Jozsa_Algorithm(oracle_gate, n)
    transpiled_Deutsch_Jozsa_Circuit = transpile(Deutsch_Jozsa_Circuit, Fake_Vigo_Sim)
    qobj = assemble(transpiled_Deutsch_Jozsa_Circuit)
    results = Fake_Vigo_Sim.run(qobj).result()
    answer = results.get_counts()

    plot_histogram(answer)
    plt.show()

main()