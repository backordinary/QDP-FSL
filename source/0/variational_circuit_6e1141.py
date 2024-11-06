# https://github.com/codope/qiskit-india-challenge/blob/bbd002ecb7bf116ab5441bf273f7996d9fe0921d/finale/variational_circuit.py
# the write_and_run function writes the content in this cell into the file "variational_circuit.py"

### WRITE YOUR CODE BETWEEN THESE LINES - START
    
# import libraries that are used in the function below.
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import  RealAmplitudes, EfficientSU2, TwoLocal, NLocal
    
### WRITE YOUR CODE BETWEEN THESE LINES - END

def variational_circuit():
    # BUILD VARIATIONAL CIRCUIT HERE - START
    
    # import required qiskit libraries if additional libraries are required
    
    # build the variational circuit
    var_circuit = EfficientSU2(3, entanglement='full', reps=3)
    # var_circuit = RealAmplitudes(3, entanglement='circular', reps=3)
    # var_circuit = TwoLocal(2, ['ry', 'rz'], 'cz', reps=3)
    # var_circuit = EfficientSU2(3, su2_gates=['rx', 'y'], entanglement='linear', reps=3)

    # BUILD VARIATIONAL CIRCUIT HERE - END
    
    # return the variational circuit which is either a VaritionalForm or QuantumCircuit object
    return var_circuit
