# https://github.com/codope/qiskit-india-challenge/blob/bbd002ecb7bf116ab5441bf273f7996d9fe0921d/finale/feature_map.py
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
