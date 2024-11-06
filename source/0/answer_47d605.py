# https://github.com/techwithshadab/QuantumComputing/blob/efe5e1ac25312d3fad4f58caa5164e8333292892/Qiskit-India-Challenge/Accuracy%2080.4/answer.py
# the write_and_run function writes the content in this cell into the file "feature_map.py"
 
### WRITE YOUR CODE BETWEEN THESE LINES - START
    
# import libraries that are used in the function below.
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap, PauliFeatureMap, NLocal
 
### WRITE YOUR CODE BETWEEN THESE LINES - END
 
def feature_map(): 
    # BUILD FEATURE MAP HERE - START
    
    # import required qiskit libraries if additional libraries are required
    
    # build the feature map
    feature_map = ZFeatureMap(feature_dimension=3, reps=2) #ZFeatureMap(feature_dimension=3, reps=2, insert_barriers=True)
    
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
    
    var_circuit = RealAmplitudes(3, entanglement='full', reps=2) #TwoLocal(3, ['ry', 'rz'], 'cz', reps=3)
 
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
    
    optimal_parameters = [ 1.42952266, -0.71424806, -0.02948316,  0.50984144,  0.30504165,
        1.00013595,  0.22629046,  0.57628436, -1.15156946]
    # STORE THE OPTIMAL PARAMETERS AS AN ARRAY IN THE VARIABLE optimal_parameters 
    return np.array(optimal_parameters)
