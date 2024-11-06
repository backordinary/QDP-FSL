# https://github.com/Hannibalcarthaga/Qiskit-Challenge-India-2020/blob/82dac8d2917170d1bd4b493d472af3dbfdc3183e/answer_day2_question1-checkpoint.py

### WRITE YOUR CODE BETWEEN THESE LINES - START
    
# import libraries that are used in the function below.
from qiskit import QuantumCircuit, execute, Aer
import numpy as np
from math import pi, sqrt
from qiskit.visualization import plot_bloch_multivector
from qiskit.extensions import Initialize
### WRITE YOUR CODE BETWEEN THESE LINES - END

def build_state():
     
    # intialized a quantum circuit on one qubit
    circuit = QuantumCircuit(1)
   
    ### WRITE YOUR CODE BETWEEN THESE LINES - START
    initial_state = [1,0]
    # apply the necessary u3 gate
    circuit.u3(2*pi/3, pi/2, 0, 0)
    ### WRITE YOUR CODE BETWEEN THESE LINES - END
    return circuit
