# https://github.com/harshitgarg22/qiskit-india-challenge-2020/blob/0aaf1ee6d6b4cb92f2416a59ccb31ce586c090ee/day_2/answer_day2_question2.py

### WRITE YOUR CODE BETWEEN THESE LINES - START
    
# import libraries that are used in the function below.
from qiskit import QuantumCircuit
import numpy as np
import math
### WRITE YOUR CODE BETWEEN THESE LINES - END

def build_state():
    
    # create a quantum circuit on one qubit
    circuit = QuantumCircuit(1)
    
    ### WRITE YOUR CODE BETWEEN THESE LINES - START
    initial_state = [0,1]
    circuit.initialize(initial_state, 0)
    circuit.ry(math.pi/3,0)
    # apply necessary gates
    
    ### WRITE YOUR CODE BETWEEN THESE LINES - END
    return circuit
