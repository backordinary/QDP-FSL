# https://github.com/mehak6569/classification-of-handwritten-digits/blob/a579ce174c06f9e2a597bb0f53b782bf67be7f0f/feature_map.py
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
    
    feature_map = ZZFeatureMap(feature_dimension=3, reps=3, entanglement='full')
    
               
    # BUILD FEATURE MAP HERE - END
    
    #return the feature map which is either a FeatureMap or QuantumCircuit object
    return feature_map
