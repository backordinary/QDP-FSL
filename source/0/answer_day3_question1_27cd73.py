# https://github.com/harshitgarg22/qiskit-india-challenge-2020/blob/0aaf1ee6d6b4cb92f2416a59ccb31ce586c090ee/day_3/answer_day3_question1.py

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
    circuit.initialize([1/sqrt(2), -1j/sqrt(2)],0)
    circuit.rz(-pi/2,0)
    circuit.h(0)
#     circuit.z(0)
#     circuit.h(0)
#     circuit.h(0)
#     circuit.z(0)
#     circuit.h(0)
    # apply necessary gates
    
    ### WRITE YOUR CODE BETWEEN THESE LINES - END
    return circuit
