# https://github.com/maniraman-periyasamy/QuantumBinaryAdder/blob/44e8c54a14f1703cddcdc3f6b32fe49eb8747c66/QuantumBinaryAdded.py
"""
This is a simple implementation of a Quantum Incrementer and addition circuit which increases any given
arbitary binary number (string format) or adds 2 aritary binaty number (string format) as implementd in the paper

 **Reversible addition circuit using one ancillary bit with application to quantum computing**

"""

from qiskit import *
import matplotlib.pyplot as plt
from qiskit import tools
from qiskit.tools.visualization import plot_histogram
from qiskit.circuit.library import MCMT, MCXGate





def increment(a):

    """
    a - binary number to be incremented by 1
    """

    qr = QuantumRegister(len(a)+1)
    cr = ClassicalRegister(len(a)+1)
    circuit = QuantumCircuit(qr,cr)
    circuit.x(qr[len(a)])

    for i, val in enumerate(a[::-1]):
        if val == "1":
            circuit.x(i)

    for i in range(len(a)):
        circuit.cx(qr[len(a)], qr[i])
        ctrlString = "1"+"0"*(i)
        tempmc = MCXGate(i+1,ctrl_state=ctrlString)
        circuit.append(tempmc,qr[:i+1]+[qr[len(a)]],[])

    circuit.x(qr[len(a)])
    ctrlString ="0"*(len(a)-1)
    tempmc = MCXGate(len(a)-1,ctrl_state=ctrlString)
    circuit.append(tempmc,qr[0:len(a)-1]+[qr[len(a)]],[])
    circuit.barrier()

    circuit.measure(qr,cr)

    #circuit.draw(output="mpl")
    #plt.show() # uncomment to print the circuit

    simulator = Aer.get_backend("qasm_simulator")
    results = execute(circuit,simulator, shots=1024).result()
    
    plot_histogram(results.get_counts(circuit))
    plt.show()


def add(a,b):

    """
    a,b - binary number to be added
    """




    # smaller number as the control qbit is required for the logic to work! (but why?)
    a_binary = ''.join(format(ord(i), 'b') for i in a)
    b_binary = ''.join(format(ord(i), 'b') for i in b)

    if a_binary<=b_binary :
        first_nuber = a
        second_number = b
    else:
        first_nuber = b
        second_number = a

    digits_in_first = len(first_nuber)
    digits_in_second = len(second_number)
    total_digits = digits_in_first+digits_in_second

  

    qrA = QuantumRegister(digits_in_first,"a")
    qrB = QuantumRegister(digits_in_second,"b")
    qrAcc = QuantumRegister(1,"acc")
    cr = ClassicalRegister(digits_in_second+1)
    circuit = QuantumCircuit(qrA,qrB,qrAcc,cr)
    circuit.x(qrAcc)

    for i, val in enumerate(first_nuber[::-1]+second_number[::-1]):
        if val == "1":
            circuit.x(i)

    for i in range(digits_in_first):
        for j in range(i,digits_in_second):
            
            ctrlString = "1"+"1"
            tempmc = MCXGate(2,ctrl_state=ctrlString)
            circuit.append(tempmc,[qrA[i]]+[qrAcc]+[qrB[j]],[])
        
            ctrlString = "1"+"0"*(j-i)+"1"
            tempmc = MCXGate(j-i+2,ctrl_state=ctrlString)
            circuit.append(tempmc,[qrA[i]]+qrB[i:j+1]+[qrAcc],[])
        circuit.x(qrAcc)
        ctrlString ="0"*(len(range(i,digits_in_second-1)))+"1"
        tempmc = MCXGate(digits_in_second-i,ctrl_state=ctrlString)
        circuit.append(tempmc,[qrA[i]]+ qrB[i:digits_in_second-1] + [qrAcc],[])
        circuit.barrier()

    circuit.measure(qrB,cr[0:digits_in_second])
    circuit.measure(qrAcc,cr[digits_in_second])
    

    #circuit.draw(output="mpl")
    #plt.tight_layout()
    #plt.show() # uncomment to print the circuit

    simulator = Aer.get_backend("qasm_simulator")
    results = execute(circuit,simulator, shots=1024).result()
    
    plot_histogram(results.get_counts(circuit))
    plt.tight_layout()
    plt.show()


# Enter the Binary numbers a and b

a = "100111"
b = "011001"

assert len(a) == len(b),"Both a and b should have same number of significant digits"

#increment(a)
add(a,b)
    