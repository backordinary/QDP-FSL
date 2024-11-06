# https://github.com/Avhijit-codeboy/My_Quantum_things/blob/7a6708f5e19ecdf1a28bee061f11017e5f505c1b/Qiskit%20India%20challenge/answer.py
# the write_and_run function writes the content in this cell into the file "feature_map.py"

### WRITE YOUR CODE BETWEEN THESE LINES - START
    
# import libraries that are used in the function below.
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap, PauliFeatureMap
import numpy as np
    
### WRITE YOUR CODE BETWEEN THESE LINES - END

def feature_map(): 
    # BUILD FEATURE MAP HERE - START
    
    # import required qiskit libraries if additional libraries are required
    
    # build the feature map
    x = ParameterVector('x', length=3)
    feature_map = QuantumCircuit(3)
    feature_map.h(0)
    feature_map.rz(x[0]*np.pi/6,0)
    feature_map.h(0)
    feature_map.rx(x[0]*np.pi/2,0)
    feature_map.h(0)
    feature_map.ry(x[0]*np.pi/6,0)
    feature_map.h(0)
    feature_map.h(1)
    feature_map.rz(x[1]*np.pi/6,1)
    feature_map.h(1)
    feature_map.rx(x[1]*np.pi/2,1)
    feature_map.h(1)
    feature_map.ry(x[1]*np.pi/6,1)
    feature_map.h(1)
    feature_map.h(2)
    feature_map.rz(x[2]*np.pi/6,2)
    feature_map.h(2)
    feature_map.rx(x[2]*np.pi/2,2)
    feature_map.h(2)
    feature_map.ry(x[2]*np.pi/6,2)
    feature_map.h(2)
    for j in range(0 + 1, 3):
        feature_map.cx(0, j)
        feature_map.u1(2*((np.pi/2-x[0]) * (np.pi/2-x[j])), j)
        feature_map.cx(0, j)
    for j in range(1 + 1, 3):
        feature_map.cx(1, j)
        feature_map.u1(2*((np.pi/2-x[1]) * (np.pi/2-x[j])), j)
        feature_map.cx(1, j)
    for j in range(2 + 1, 3):
        feature_map.cx(2, j)
        feature_map.u1(2*((np.pi/2-x[2]) * (np.pi/2-x[j])), j)
        feature_map.cx(2, j)
    feature_map.h(0)
    feature_map.rz(x[0]*np.pi/6,0)
    feature_map.h(0)
    feature_map.rx(x[0]*np.pi/2,0)
    feature_map.h(0)
    feature_map.ry(x[0]*np.pi/6,0)
    feature_map.h(0)
    feature_map.h(1)
    feature_map.rz(x[1]*np.pi/6,1)
    feature_map.h(1)
    feature_map.rx(x[1]*np.pi/2,1)
    feature_map.h(1)
    feature_map.ry(x[1]*np.pi/6,1)
    feature_map.h(1)
    feature_map.h(2)
    feature_map.rz(x[2]*np.pi/6,2)
    feature_map.h(2)
    feature_map.rx(x[2]*np.pi/2,2)
    feature_map.h(2)
    feature_map.ry(x[2]*np.pi/6,2)
    feature_map.h(2)
    for j in range(0 + 1, 3):
        feature_map.cx(0, j)
        feature_map.u1(2*((np.pi/2-x[0]) * (np.pi/2-x[j])), j)
        feature_map.cx(0, j)
    for j in range(1 + 1, 3):
        feature_map.cx(1, j)
        feature_map.u1(2*((np.pi/2-x[1]) * (np.pi/2-x[j])), j)
        feature_map.cx(1, j)
    for j in range(2 + 1, 3):
        feature_map.cx(2, j)
        feature_map.u1(2*((np.pi/2-x[2]) * (np.pi/2-x[j])), j)
        feature_map.cx(2, j)
    
    # BUILD FEATURE MAP HERE - END
    
    #return the feature map which is either a FeatureMap or QuantumCircuit object
    return feature_map
# the write_and_run function writes the content in this cell into the file "variational_circuit.py"

### WRITE YOUR CODE BETWEEN THESE LINES - START
    
# import libraries that are used in the function below.
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import  RealAmplitudes, EfficientSU2
    
### WRITE YOUR CODE BETWEEN THESE LINES - END

def variational_circuit():
    # BUILD VARIATIONAL CIRCUIT HERE - START
    
    # import required qiskit libraries if additional libraries are required
    
    # build the variational circuit
    var_circuit = EfficientSU2(3, reps=3)

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
    
    optimal_parameters = [ 1.53366614, -1.180565  , -0.83242541,  1.42514794, -0.6539459 ,
        0.75997613, -0.46002519, -1.04103125,  0.17166766, -0.39573853,
        3.62113631, -0.5200239 ,  0.71020846,  1.31452177,  0.57627883,
       -1.70502093,  0.98979859,  2.13270941, -0.75832064, -0.20020705,
       -0.31182005, -1.04739694, -1.03982196,  1.47795992]
    
    # STORE THE OPTIMAL PARAMETERS AS AN ARRAY IN THE VARIABLE optimal_parameters 
    return np.array(optimal_parameters)
