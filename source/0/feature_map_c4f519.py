# https://github.com/Avhijit-codeboy/My_Quantum_things/blob/7a6708f5e19ecdf1a28bee061f11017e5f505c1b/Qiskit%20India%20challenge/feature_map.py
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
