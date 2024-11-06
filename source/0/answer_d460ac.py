# https://github.com/mehak6569/classification-of-handwritten-digits/blob/a579ce174c06f9e2a597bb0f53b782bf67be7f0f/answer.py
# the write_and_run function writes the content in this cell into the file "feature_map.py"

### WRITE YOUR CODE BETWEEN THESE LINES - START
    
# import libraries that are used in the function below.
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap, PauliFeatureMap
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    
### WRITE YOUR CODE BETWEEN THESE LINES - END

def feature_map(): 
    # BUILD FEATURE MAP HERE - START
    
    # import required qiskit libraries if additional libraries are required
    
    
    np.random.seed(0)

    weights = np.random.rand(2,2,3) # 2 qubits, 2 layers, 3 weights in each layer

    qc= QuantumCircuit(2)

    qc.u3(training_set[0][0],training_set[0][1],0,0)
    qc.u3(training_set[0][0],training_set[0][1],0,1)

    qc.u3(weights[0][0][0],weights[0][0][1],weights[0][0][2],0)
    qc.u3(weights[1][0][0],weights[1][0][1],weights[1][0][2],1)

    qc.cz(0,1)

    qc.u3(training_set[0][0],training_set[0][1],0,0)
    qc.u3(training_set[0][0],training_set[0][1],0,1)

    qc.u3(weights[0][1][0],weights[0][1][1],weights[0][1][2],0)
    qc.u3(weights[1][1][0],weights[1][1][1],weights[1][1][2],1)
    #num_qubits = 3            
    #reps = 1              # number of times you'd want to repeat the circuit

    #x = ParameterVector('x', length=num_qubits)  # creating a list of Parameters
    #feature_map = QuantumCircuit(num_qubits)

    # defining our parametric form
    #for _ in range(reps):
    #   for i in range(num_qubits):
     #       feature_map.rx(x[i], i)
     #   for i in range(num_qubits):
     #       for j in range(i + 1, num_qubits):
     #           feature_map.cx(i,j)
    
    # BUILD FEATURE MAP HERE - END
    
    #return the feature map which is either a FeatureMap or QuantumCircuit object
    return feature_map
# the write_and_run function writes the content in this cell into the file "variational_circuit.py"

### WRITE YOUR CODE BETWEEN THESE LINES - START
    
# import libraries that are used in the function below.
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import  RealAmplitudes, EfficientSU2, TwoLocal

### WRITE YOUR CODE BETWEEN THESE LINES - END

def variational_circuit():
    # BUILD VARIATIONAL CIRCUIT HERE - START
    
    # import required qiskit libraries if additional libraries are required
    
    # build the variational circuit
    
    var_circuit = EfficientSU2(3, entanglement='full', reps=4)


    # BUILD VARIATIONAL CIRCUIT HERE - END
   
    # return the variational circuit which is either a VaritionalForm or QuantumCircuit object
    return var_circuit

# # the write_and_run function writes the content in this cell into the file "optimal_params.py"

### WRITE YOUR CODE BETWEEN THESE LINES - START
    
# import libraries that are used in the function below.
import numpy as np
    
### WRITE YOUR CODE BETWEEN THESE LINES - END

def return_optimal_params():
    # STORE THE OPTIMAL PARAMETERS AS AN ARRAY IN THE VARIABLE optimal_parameters 
    
    optimal_parameters =[ 1.23794261e+00, -6.05284490e-01, -8.45595458e-01, -6.15462673e-01,
       -2.11647697e+00, -1.28206956e-01,  2.69102247e-01, -8.05780497e-01,
       -1.61144291e+00,  1.19445826e+00, -6.54716349e-01,  1.67773617e+00,
       -8.38864624e-01,  1.68450798e+00,  1.55359638e-01, -1.11879663e+00,
       -8.40470191e-01,  1.81922255e+00,  4.32943964e-01,  6.24630427e-01,
        8.18335290e-01, -2.75627849e-02,  5.78596149e-01,  1.65019934e-03,
       -7.22121555e-01, -4.09009221e-01,  5.49457358e-02,  1.21729358e+00,
       -3.30963915e-01,  5.09766280e-01]
    # STORE THE OPTIMAL PARAMETERS AS AN ARRAY IN THE VARIABLE optimal_parameters 
    return np.array(optimal_parameters)
