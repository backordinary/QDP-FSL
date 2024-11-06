# https://github.com/harshagarine/QISKIT_INDIA_CHALLENGE/blob/53d83be9c106013b64bf1ff2ee649c87fd05c8c0/answer_day3_question1.py

### WRITE YOUR CODE BETWEEN THESE LINES - START
    
# import libraries that are used in the function below.
from qiskit import QuantumCircuit
import numpy as np
from math import sqrt, pi
    
### WRITE YOUR CODE BETWEEN THESE LINES - END
def build_state():

    # create a quantum circuit on one qubit
    circuit = QuantumCircuit(1)

    ### WRITE YOUR CODE BETWEEN THESE LINES - START
    init = [1/sqrt(2),0-1j/sqrt(2)]
    circuit.initialize(init,0)
    # apply necessary gates
    circuit.rx(pi/2,0)
    ### WRITE YOUR CODE BETWEEN THESE LINES - END
    return circuit
