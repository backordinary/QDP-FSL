# https://github.com/huwenqing0606/Quantum-Computing-using-Qiskit/blob/bc0f21798346760ad12ed349ed35ba411f1efdb9/QuantumFourierTransform.py
import numpy as np
from numpy import pi
# importing Qiskit
from qiskit import QuantumCircuit, transpile, assemble, Aer, IBMQ
from qiskit.providers.ibmq import least_busy
from qiskit.tools.monitor import job_monitor
# import basic plot tools
from qiskit.visualization import plot_histogram, plot_bloch_multivector
import matplotlib.pyplot as plt


# demonstrate a simple 4 qubit Quantum Fourier Transform (QFourierT) circuit
def QFourier_demonstrate(qc_init):
    
    qc = QuantumCircuit(4)

    qc += qc_init
    qc.barrier()
    qc.draw(output='mpl')
    plt.show()

    qc.h(3)
    qc.draw(output='mpl')
    plt.show()

    qc.cp(pi/2, 3, 2) # CROT with control qubit 2 and target qubit 3
    qc.cp(pi/4, 3, 1) # CROT with control qubit 1 and target qubit 3
    qc.cp(pi/8, 3, 0) # CROT with control qubit 0 and target qubit 3 
    qc.barrier()
    qc.draw(output='mpl')
    plt.show()

    qc.h(2)
    qc.cp(pi/2, 2, 1) # CROT with control qubit 1 and target qubit 2
    qc.cp(pi/4, 2, 0) # CROT with control qubit 0 and target qubit 2
    qc.barrier()

    qc.h(1)
    qc.cp(pi/2, 1, 0) # CROT with control qubit 0 and target qubit 1
    qc.barrier()
    qc.h(0)
    qc.barrier()
    qc.draw(output='mpl')
    plt.show()

    qc.swap(0,3)
    qc.swap(1,2)
    qc.draw(output='mpl')   
    plt.show()

    return qc


# recursively build the sequence of H and CROT gates for QFourierT circuit 
# at the first n qubits of circuit
def QFourier_H_CROT(circuit, target_qubit, n):
    if target_qubit==-1:
        return circuit
    circuit.h(target_qubit)
    for m in range(target_qubit):
        control_qubit = target_qubit-m-1
        circuit.cp(pi/2**(target_qubit-control_qubit), control_qubit, target_qubit)
    circuit.barrier()

    QFourier_H_CROT(circuit, target_qubit-1, n)


# the swap gate of Quantum Fourier Transform 
def QFourier_swap(circuit, n):
    for qubit in range(n//2):
        circuit.swap(qubit, n-qubit-1)
    return circuit


# build the QFourierT circuit in the first n qubits of qc_init
def QFourier(circuit, n):
    # add the H and CROT gates
    QFourier_H_CROT(circuit, n-1, n)
    # add the swap gate
    QFourier_swap(circuit, n)

    return circuit


# initialize the QFourierT circuit
# n is number of registers, j is the initial state in [0, 2**n-1]
def QFourier_init(n, j):
    if (j<0 or j>2**n):
        print('j out of range: 0 to 2**n\n')
        return None
    # turn j into binary, highest power of 2 on right
    init_state = bin(j)[2:].zfill(n)[::-1]
    # Encode the initial state
    qc_init = QuantumCircuit(n)
    for qubit in range(n):
        if init_state[qubit]=='1':
            qc_init.x(qubit)
    qc_init.barrier()

    return qc_init


# inverse Quantum Fourier Transform on the first n qubits of qc_final
# non-recursive form
def QFourier_inverse(circuit, n):
    # the Swaps
    QFourier_swap(circuit, n)
    circuit.barrier()
    # the H and CROT gates
    for target_qubit in range(n):
        for control_qubit in range(target_qubit):
            circuit.cp(-pi/2**(target_qubit-control_qubit), control_qubit, target_qubit)
        circuit.h(target_qubit)
        circuit.barrier()
    return circuit


# show the circuit construction of Quantum Fourier Transform and the state vector change
# also show inverse transform
# if demonstrate = 1, show a demonstration of a 4 qubit circuit
# n is number of registers, j is the initial state in [0, 2**n-1]
def QFourier_showcircuit(demonstrate, n, j):
    if demonstrate:
        # Encode the initial state
        qc_init = QuantumCircuit(4)
        qc_init.x(0)
        qc_init.x(2)
        # build the Quantum Fourier Transform circuit for demonstration
        qc=QFourier_demonstrate(qc_init)
        # check inverse transform
        qc_inverse = qc.copy()
        QFourier_inverse(qc_inverse, 4)
        qc_inverse.draw(output='mpl')
        plt.show()        
    else:
        # Encode the initial state
        qc_init = QFourier_init(n, j)
        qc_init.draw(output='mpl')
        plt.show()
        # build the Quantum Fourier Transform circuit for demonstration
        qc = qc_init.copy()
        QFourier(qc, n)
        qc.draw(output='mpl')
        plt.show()
        # check inverse transform
        qc_final = qc.copy()
        qc_inverse = QFourier_inverse(qc_final, n)
        qc_inverse.draw(output='mpl')
        plt.show()

        
    # start plotting the change of basis in Quantum Fourier Transform
    sim = Aer.get_backend("aer_simulator")
    # the original basis
    qc_init.save_statevector()
    statevector = sim.run(qc_init).result().get_statevector()
    plot_bloch_multivector(statevector)
    plt.show()
    # the transformed basis
    qc.save_statevector()
    statevector = sim.run(qc).result().get_statevector()
    plot_bloch_multivector(statevector)
    #plt.savefig('D:\\Temporary Files\\Quantum Computing_2021_SummerSeminar\\qiskit_code\\QFourier_'+str(j))
    plt.show()
    # the inverse transformed basis
    qc_inverse.save_statevector()
    statevector = sim.run(qc_inverse).result().get_statevector()
    plot_bloch_multivector(statevector)
    plt.show()

    return None


# produce a sequence of figures showing the change of Fourier basis with n input qubits
def QFourier_produceanimation(animate, n):

    if not animate:
        return None
    
    for j in range(2**n-1):
        # Encode the initial state
        qc = QFourier_init(n, j)
        # build the Quantum Fourier Transform circuit for demonstration
        QFourier(qc, n)

        # start plotting the change of basis in Quantum Fourier Transform
        sim = Aer.get_backend("aer_simulator")
        # the transformed basis
        qc.save_statevector()
        statevector = sim.run(qc).result().get_statevector()
        plot_bloch_multivector(statevector)
        plt.savefig('D:\\Temporary Files\\Quantum Computing_2021_SummerSeminar\\qiskit_code\\QFourier_'+str(j))

    return None


if __name__=='__main__':

    # demonstrate chooses to demonstrate or not, =1 means demonstrate
    # n is number of registers, j is the initial state in [0, 2**n-1]
    QFourier_showcircuit(demonstrate=0, n=5, j=31)
    # animate chooses to produce sequence of animated basis or not
    # produce a sequence of figures showing the change of Fourier basis with n input qubits
    #QFourier_produceanimation(animate=1, n=5)
