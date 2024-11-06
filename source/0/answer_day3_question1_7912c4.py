# https://github.com/paripooranan/Qiskit-India-Challenge/blob/7c016f85c9169c52f33c3d0a7a2f200477d2415d/daily%20challenges/answer_day3_question1.py

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
    init = [1/sqrt(2)*1,-1j*1/sqrt(2)]
    circuit.initialize(init,0)
    # apply necessary gates
    circuit.rx(pi/2,0)
    ### WRITE YOUR CODE BETWEEN THESE LINES - END
    return circuit
