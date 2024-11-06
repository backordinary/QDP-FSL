# https://github.com/huwenqing0606/Quantum-Computing-using-Qiskit/blob/81d4307c1d7b23f5e7c8806fe013ea51ee400d50/Grover.py
import numpy as np
# importing Qiskit
from qiskit import IBMQ, Aer, assemble, transpile
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.providers.ibmq import least_busy
# import basic plot tools
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram


# construct a Grover's circuit on n qubits
# given n-qubit oracle U_omega and the Grover's diffusion operator U_s
# repeat U_sU_omega given number of times (t)
def Grover_circuit(n, oracle, diffuser, t):
    # initialization
    circuit = QuantumCircuit(n)
    # add H gates
    for qubit in range(n):
        circuit.h(qubit)
    circuit.barrier()
    # repeat t times of U_sU_omega
    for iteration in range(t):
        circuit += oracle
        circuit += diffuser
    # return Grover's circuit
    return circuit


# the oracle U_omega for Grover's circuit with omega = |11> 
def Grover_oracle_11():
    circuit = QuantumCircuit(2)
    # add CZ gates
    circuit.cz(0, 1)
    # add barrier
    circuit.barrier()
    return circuit


# the oracle U_omega for Grover's circuit with omega = |101> or |110>
# See C. Figgatt et al, "Complete 3-Qubit Grover search on a programmable quantum computer", 
#   Nature Communications, Vol 8, Art 1918, 2017, doi:10.1038/s41467-017-01904-7, arXiv:1703.10535 
def Grover_oracle_101_110():
    circuit = QuantumCircuit(3)
    # add CZ gates
    circuit.cz(0, 2)
    circuit.cz(1, 2)
    # add barrier
    circuit.barrier()
    return circuit


# the general Grover's oracle with given binary string b
def Grover_oracle(b):
    b = str(b)
    n = len(b)
    circuit = QuantumCircuit(n)
    # Do multi-controlled-Z gate
    circuit.h(n-1)
    circuit.mct(list(range(n-1)), n-1)  # multi-controlled-toffoli
    circuit.h(n-1)
    circuit.barrier()
    # flip according to b, use not gates
    for index in range(n):
        if b[n-1-index]=='0':
            circuit.x(index)
    circuit.barrier()
    return circuit


# the diffuser U_s for Grover's circuit on 2 qubits
def Grover_diffuser_2qubits():
    circuit = QuantumCircuit(2)
    # add H gates
    circuit.h([0,1])
    # add Z gates
    circuit.z([0,1])
    # add CZ gates
    circuit.cz(0,1)
    # add H gates
    circuit.h([0,1])
    # add barrier
    circuit.barrier()
    return circuit


# the general Grover's diffuser for n qubits
def Grover_diffuser(n):
    circuit = QuantumCircuit(n)
    # Apply H-gates
    for qubit in range(n):
        circuit.h(qubit)
    circuit.barrier()
    # Apply X-gates
    for qubit in range(n):
        circuit.x(qubit)
    circuit.barrier()
    # Do multi-controlled-Z gate
    circuit.h(n-1)
    circuit.mct(list(range(n-1)), n-1)  # multi-controlled-toffoli
    circuit.h(n-1)
    circuit.barrier()
    # Apply X-gates
    for qubit in range(n):
        circuit.x(qubit)
    circuit.barrier()
    # Apply H-gates
    for qubit in range(n):
        circuit.h(qubit)
    circuit.barrier()
    return circuit


if __name__=='__main__':
    # two choices: (1) 2 qubits, omega = |11>; (2) 3 qubits, omega = |101> and |110>
    # set oracle and diffuser
    demonstrate_11 = 0
    demonstrate_101_110 = 0
    show = 0
    guess = 1
    if demonstrate_11:
        n = 2 
        oracle = Grover_oracle_11()
        diffuser = Grover_diffuser_2qubits()
    elif demonstrate_101_110:
        n = 3
        oracle = Grover_oracle_101_110()
        diffuser = Grover_diffuser(n)
    else:
        b = '101110010101'
        n = len(b)
        oracle = Grover_oracle(b)
        diffuser = Grover_diffuser(n)

    c = np.pi/4
    t = int(c*np.sqrt(n))
    Grover_circuit = Grover_circuit(n, oracle, diffuser, t)
    Grover_circuit.draw(output='mpl')
    if show:
        plt.show()

    # simulate measurement
    Grover_circuit.measure_all()
    aer_sim = Aer.get_backend('aer_simulator')
    shots = 5*(2**n)
    qobj = assemble(Grover_circuit, shots=shots)
    result = aer_sim.run(qobj).result()
    counts = result.get_counts()
    plot_histogram(counts)
    if show:
        plt.show()
    # find the maximal count string as the Grover guess result
    if guess:
        maximal_count = 0
        Grover_guess = ''
        for key in counts.keys():
            print(key, ':', counts[key])
            if counts[key]>=maximal_count:
                Grover_guess = key
                maximal_count = counts[key]
        print('Grover guess is', Grover_guess, 'count is', maximal_count)
        if Grover_guess == b:
            print('Right Grover guess :)')
        else:
            print('Wrong Grover guess :(')
