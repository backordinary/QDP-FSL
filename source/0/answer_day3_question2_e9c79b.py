# https://github.com/Avhijit-codeboy/My_Quantum_things/blob/7a6708f5e19ecdf1a28bee061f11017e5f505c1b/Qiskit%20India%20challenge/answer_day3_question2.py

### WRITE YOUR CODE BETWEEN THESE LINES - START
    
# import libraries that are used in the function below.
from qiskit import QuantumCircuit
import numpy as np
    
### WRITE YOUR CODE BETWEEN THESE LINES - END

def build_state():
    
    # create a quantum circuit on two qubits
    circuit = QuantumCircuit(2)
    circuit.x(0)
    circuit.x(1)
    circuit.h(1)
    ### WRITE YOUR CODE BETWEEN THESE LINES - START
    
    # apply necessary gates
    circuit.cx(1,0)
    ### WRITE YOUR CODE BETWEEN THESE LINES - END
    return circuit
