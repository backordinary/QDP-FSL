# https://github.com/kthod/VQLS_algorithm/blob/89f5012e7547a5c41b0101797e93cf53616fddfa/circuits.py
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import math
import numpy as np
from matplotlib import pyplot as plt

def control_A_l(circ,A_l,ancilla, qubits):
    """
    circuit to control Al
    using the ancilla 
    """
    for i in range (0, len(A_l)):
        if (A_l[i] == 1):
             circ.cz(ancilla, qubits[i])
        if (A_l[i] == 2):
             circ.cx(ancilla, qubits[i])
        if (A_l[i] == 3):
             circ.ch(ancilla, qubits[i])

    
def ansatz(circ,qubits, parameters):
    """
    circuit to implemenet the ansatz
    """
    for i in range (0, len(qubits)):
        circ.ry(parameters[0][i], qubits[i])


    for i in range (0, len(qubits)-1):
        circ.cx(qubits[i], qubits[i+1])
   

    for i in range (0, len(qubits)):
        circ.ry(parameters[1][i], qubits[i])

    for i in range (0, len(qubits)-1):
        circ.cx(qubits[i], qubits[i+1])
   

    for iz in range (0, len(qubits)):
        circ.ry(parameters[2][iz], qubits[iz])





def had_test(circ,qubits,A_l1,A_l2, ancilla, parameters):
    """
    Hadamard test for beta_ll' terms
    """
    circ.h(ancilla)

    ansatz(circ,qubits, parameters)


    control_A_l(circ,A_l1,ancilla, qubits)
    
    control_A_l(circ,A_l2,ancilla, qubits)
    
    circ.h(ancilla)




def control_ansatz(circ,qubits, parameters, ancilla):
    """
    circuit to control the ansatz
    """

    for i in range (0, len(qubits)):
        circ.cry(parameters[0][i],  ancilla, qubits[i])

    for i in range (0, len(qubits)-1):
        circ.ccx(ancilla,qubits[i], qubits[i+1])

    


    for i in range (0, len(qubits)):
        circ.cry(parameters[1][i],  ancilla, qubits[i])


    for i in range (0, len(qubits)-1):
        circ.ccx(ancilla,qubits[i], qubits[i+1])
    

    for i in range (0, len(qubits)):
        circ.cry(parameters[2][i],  ancilla,  qubits[i])






def control_b(circ,ancilla, qubits):

    for ia in qubits:
        circ.ch(ancilla, ia)








def special_had_test(circ, A_l, qubits, ancilla, parameters):
    """
    Hadamard test for gamma_ll' terms
    """
    circ.h(ancilla)

    control_ansatz(circ,qubits, parameters, ancilla)

    control_A_l(circ,A_l,ancilla, qubits)
    


    control_b(circ,ancilla, qubits)
    
    circ.h(ancilla)

    #circ.measure(ancilla,reg)



# circ = QuantumCircuit(3)
# ansatz(circ,[0, 1, 2], [[1, 1, 1], [1, 1, 1], [1, 1, 1]])
# circ.draw(output="mpl")

# circ = QuantumCircuit(4)
# had_test(circ, [1, 2, 3],[1, 0, 0], [0, 0, 2], 0, [[1, 1, 1], [1, 1, 1], [1, 1, 1]])
# circ.draw(output="mpl")

# circ = QuantumCircuit(4)
# special_had_test(circ, [1, 0, 0], [1, 2, 3], 0, [[1, 1, 1], [1, 1, 1], [1, 1, 1]])
# circ.draw(output="mpl")
# plt.show()