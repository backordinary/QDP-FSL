# https://github.com/paripooranan/Qiskit-India-Challenge/blob/7c016f85c9169c52f33c3d0a7a2f200477d2415d/daily%20challenges/answer_day2_question2.py

### WRITE YOUR CODE BETWEEN THESE LINES - START
    
# import libraries that are used in the function below.
from qiskit import QuantumCircuit
import numpy as np
    
### WRITE YOUR CODE BETWEEN THESE LINES - END

def build_state():
    
    # create a quantum circuit on one qubit
    circuit = QuantumCircuit(1)
    
    ### WRITE YOUR CODE BETWEEN THESE LINES - START
    init = [0,1]
    circuit.initialize(init,0)
    circuit.ry(np.pi/3,0)
    # apply necessary gates
    
    ### WRITE YOUR CODE BETWEEN THESE LINES - END
    return circuit
