# https://github.com/paripooranan/Qiskit-India-Challenge/blob/7c016f85c9169c52f33c3d0a7a2f200477d2415d/feature_map.py
# the write_and_run function writes the content in this cell into the file "feature_map.py"

### WRITE YOUR CODE BETWEEN THESE LINES - START
    
# import libraries that are used in the function below.
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap, PauliFeatureMap
    
### WRITE YOUR CODE BETWEEN THESE LINES - END

def feature_map(): 
    # BUILD FEATURE MAP HERE - START
    
    # import required qiskit libraries if additional libraries are required
    
    # build the feature map
    feature_dim = 3     # equal to the dimension of the data

    feature_map = ZZFeatureMap(feature_dimension=feature_dim, reps=4)
    feature_map.draw()
    # BUILD FEATURE MAP HERE - END
    
    #return the feature map which is either a FeatureMap or QuantumCircuit object
    return feature_map
