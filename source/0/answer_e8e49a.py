# https://github.com/codope/qiskit-india-challenge/blob/bbd002ecb7bf116ab5441bf273f7996d9fe0921d/finale/answer.py
# the write_and_run function writes the content in this cell into the file "feature_map.py"

### WRITE YOUR CODE BETWEEN THESE LINES - START
    
# import libraries that are used in the function below.
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap, PauliFeatureMap
from qiskit.aqua.components.feature_maps import RawFeatureVector
    
### WRITE YOUR CODE BETWEEN THESE LINES - END

def feature_map(): 
    # BUILD FEATURE MAP HERE - START
#     def custom_data_map_func(x):
#         import functools

#         coeff = x[0] if len(x) == 1 else functools.reduce(lambda m, n: m * n, np.pi - x)
#         return coeff
    
    # build the feature map
    feature_map = ZZFeatureMap(feature_dimension=3, reps=3, entanglement='full')
    # feature_map = PauliFeatureMap(feature_dimension=3, reps=3, entanglement='linear', paulis=['Z', 'YY', 'ZXZ'])
    # feature_map = RawFeatureVector(feature_dimension=3)
    
    # BUILD FEATURE MAP HERE - END
    
    #return the feature map which is either a FeatureMap or QuantumCircuit object
    return feature_map
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
# # the write_and_run function writes the content in this cell into the file "optimal_params.py"

### WRITE YOUR CODE BETWEEN THESE LINES - START
    
# import libraries that are used in the function below.
import numpy as np
    
### WRITE YOUR CODE BETWEEN THESE LINES - END

def return_optimal_params():
    # STORE THE OPTIMAL PARAMETERS AS AN ARRAY IN THE VARIABLE optimal_parameters 
    
    optimal_parameters = [ 1.5722112 ,  2.33422954,  1.80080672,  1.48378798,  0.8686183 ,
        0.59302393, -0.38912306, -0.32655619, -0.82799027,  2.68384325,
        0.99566385,  0.19888568, -0.60315524, -0.15672294, -1.12675526,
        1.04566854,  1.29105772, -0.71151711,  0.66421101, -1.75703185,
       -1.37450074,  2.33150824,  0.18538038,  1.06463035]
    
    # STORE THE OPTIMAL PARAMETERS AS AN ARRAY IN THE VARIABLE optimal_parameters 
    return np.array(optimal_parameters)
