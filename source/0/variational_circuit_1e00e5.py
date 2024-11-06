# https://github.com/mehak6569/classification-of-handwritten-digits/blob/a579ce174c06f9e2a597bb0f53b782bf67be7f0f/variational_circuit.py
# the write_and_run function writes the content in this cell into the file "variational_circuit.py"

### WRITE YOUR CODE BETWEEN THESE LINES - START
    
# import libraries that are used in the function below.
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import  RealAmplitudes, EfficientSU2, TwoLocal, ExcitationPreserving
from math import pi
from qutip import *
from qutip.qip.circuit import *
from qutip.qip.qubits import *

### WRITE YOUR CODE BETWEEN THESE LINES - END

def variational_circuit():
    # BUILD VARIATIONAL CIRCUIT HERE - START
    
    # import required qiskit libraries if additional libraries are required
    
    # build the variational circuit
    
    var_circuit = QubitCircuit(4)
    var_circuit.add_gate("SNOT", 2)

    var_circuit.add_gate("SNOT", 0)
    var_circuit.add_gate("SNOT", 1)
    var_circuit.add_gate("F1", 2, 0)
    var_circuit.add_gate("F2", 2, 1)
    var_circuit.add_gate("SWAP", [1,0])
    var_circuit.add_gate("SNOT", 1)
    var_circuit.add_gate("SINV", 0, 1)
    var_circuit.add_gate("SNOT", 0)
    var_circuit.add_gate("RY4", 3, 1)
    var_circuit.add_gate("RY8", 3, 0)
    var_circuit.add_gate("SNOT", 0)
    var_circuit.add_gate("S", 0, 1)
    var_circuit.add_gate("SNOT", 1)
    var_circuit.add_gate("SWAP", [1,0])
    var_circuit.add_gate("F2INV", 2, 1)
    var_circuit.add_gate("F1INV", 2, 0)
    var_circuit.add_gate("SNOT", 0)
    var_circuit.add_gate("SNOT", 1)

    var_circuit.add_gate("SIGMAX", 2)
    var_circuit.add_gate("RYT1", 1, 2)
    var_circuit.add_gate("CNOT", 2, 3)
    var_circuit.add_gate("RYT1d", 1, 2)    
    
    # BUILD VARIATIONAL CIRCUIT HERE - END
   
    # return the variational circuit which is either a VaritionalForm or QuantumCircuit object
    return var_circuit

