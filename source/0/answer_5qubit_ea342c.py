# https://github.com/paripooranan/Qiskit-India-Challenge/blob/7c016f85c9169c52f33c3d0a7a2f200477d2415d/answer_5qubit.py
# the write_and_run function writes the content in this cell into the file "feature_map.py"

### WRITE YOUR CODE BETWEEN THESE LINES - START
    
# import libraries that are used in the function below.
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap, PauliFeatureMap


    


def feature_map(): 
    # BUILD FEATURE MAP HERE - START
    
    # import required qiskit libraries if additional libraries are required
    
    # build the feature map

    def self_product(x: np.ndarray) -> float:
      """
      Define a function map from R^n to R.

      Args:
        x: data

      Returns:
        float: the mapped value
      """
      coeff = x[0] if len(x) == 1 else \
        functools.reduce(lambda m, n:np.square( m * n), np.pi - x)
      return coeff

    feature_dim = 5     # equal to the dimension of the data

    feature_map = ZFeatureMap(feature_dimension=feature_dim,data_map_func=self_product, reps=2)
    feature_map.draw()
    # BUILD FEATURE MAP HERE - END
    
    #return the feature map which is either a FeatureMap or QuantumCircuit object
    return feature_map# the write_and_run function writes the content in this cell into the file "variational_circuit.py"

### WRITE YOUR CODE BETWEEN THESE LINES - START
    
# import libraries that are used in the function below.
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import  RealAmplitudes, EfficientSU2
    
### WRITE YOUR CODE BETWEEN THESE LINES - END

def variational_circuit():
    # BUILD VARIATIONAL CIRCUIT HERE - START
    
    # import required qiskit libraries if additional libraries are required
    num_qubits = 5
    # build the variational circuit
    var_circuit = RealAmplitudes(num_qubits, entanglement='full', reps=3)
    var_circuit.draw()
    # BUILD VARIATIONAL CIRCUIT HERE - END
    
    # return the variational circuit which is either a VaritionalForm or QuantumCircuit object
    return var_circuit# # the write_and_run function writes the content in this cell into the file "optimal_params.py"

### WRITE YOUR CODE BETWEEN THESE LINES - START
    
# import libraries that are used in the function below.
import numpy as np

    
### WRITE YOUR CODE BETWEEN THESE LINES - END

def return_optimal_params():
    # STORE THE OPTIMAL PARAMETERS AS AN ARRAY IN THE VARIABLE optimal_parameters 
    
    optimal_parameters = [ 2.2861203 , -0.07309373, -0.35468052,  0.73175149,  1.18321584,
       -1.51842114,  0.06584612,  1.49799433,  0.69891621, -0.18451223,
        2.96230107,  0.26290385,  0.34904964, -0.36422693, -0.73480378,
        0.48140797,  1.74178837, -0.96167106,  0.92318638,  1.05062683]
    
    # STORE THE OPTIMAL PARAMETERS AS AN ARRAY IN THE VARIABLE optimal_parameters 
    return np.array(optimal_parameters)