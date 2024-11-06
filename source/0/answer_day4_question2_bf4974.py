# https://github.com/harshagarine/QISKIT_INDIA_CHALLENGE/blob/53d83be9c106013b64bf1ff2ee649c87fd05c8c0/answer_day4_question2.py

### WRITE YOUR CODE BETWEEN THESE LINES - START
    
# import libraries that are used in the functions below.
from qiskit import QuantumCircuit
import numpy as np
    
### WRITE YOUR CODE BETWEEN THESE LINES - END

def init_circuit():
    
    # create a quantum circuit on two qubits
    qc = QuantumCircuit(2)

    # initializing the circuit
    qc.h(0)
    qc.x(1)
    return qc

# The initial state has been defined above. 
# You'll now have to apply necessary gates in the build_state() function to convert the state as asked in the question.

def build_state():
    
    ### WRITE YOUR CODE BETWEEN THESE LINES - START
    
    # the initialized circuit
    circuit = init_circuit()
    
    # apply a single cu3 gate
    circuit.cu3(0, -np.pi/2, 0, 0, 1)
    
    ### WRITE YOUR CODE BETWEEN THESE LINES - END
    return circuit
