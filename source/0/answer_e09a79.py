# https://github.com/CodieKev/Variational_Quantum_Classifier_CKT/blob/a10dab69a2434ed6c9bd701fec6c13311b6d7b9f/Custom_4_Feature_Mapping_Custom_Classifier_v1/answer.py
from qiskit.aqua.components.feature_maps import FeatureMap
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import BlueprintCircuit
import numpy as np
import matplotlib.pyplot as plt
import functools

from qiskit import BasicAer
from qiskit.circuit.library import ZFeatureMap,ZZFeatureMap, PauliFeatureMap
from qiskit.aqua import QuantumInstance
from qiskit.aqua.components.feature_maps import self_product
from qiskit.aqua.algorithms import QSVM
from qiskit.ml.datasets import ad_hoc_data
from numpy import pi
class CustomFeatureMap(FeatureMap):
    """Mapping data with a custom feature map."""
    
    def __init__(self, feature_dimension, depth=2, entangler_map=None):
        """
        Args:
            feature_dimension (int): number of features
            depth (int): the number of repeated circuits
            entangler_map (list[list]): describe the connectivity of qubits, each list describes
                                        [source, target], or None for full entanglement.
                                        Note that the order is the list is the order of
                                        applying the two-qubit gate.        
        """
        self._support_parameterized_circuit = False
        self._feature_dimension = feature_dimension
        self._num_qubits = self._feature_dimension = feature_dimension+1
        self._depth = depth
        self._entangler_map = None
        if self._entangler_map is None:
            self._entangler_map = [[i, j] for i in range(self._feature_dimension) for j in range(i + 1, self._feature_dimension)]
            
    def construct_circuit(self, x, qr, inverse=False):
        """Construct the feature map circuit.
        
        Args:
            x (numpy.ndarray): 1-D to-be-transformed data.
            qr (QauntumRegister): the QuantumRegister object for the circuit.
            inverse (bool): whether or not to invert the circuit.
            
        Returns:
            QuantumCircuit: a quantum circuit transforming data x.
        """
        qc1 = QuantumCircuit(4)

        
        for _ in range(self._depth):
            y = -1.3*x[0]+x[1]
            qc1.h(0)
            qc1.h(1)
            qc1.h(2)
            qc1.h(3)
            qc1.u1(x[0],0)
            qc1.u1(x[1],1)
            qc1.u1(x[2],2)
            qc1.u1(y,3)
            qc1.cx(0,1)
            qc1.u1((2*(pi-x[0])*(pi-x[1])),1)
            qc1.cx(0,1)
            qc1.cx(0,2)
            qc1.u1((2*(pi-x[0])*(pi-x[2])),2)
            qc1.cx(0,2)
            qc1.cx(0,3)
            qc1.u1((2*(pi-x[0])*(pi-y)),3)
            qc1.cx(0,3)
            qc1.cx(1,2)
            qc1.u1((2*(pi-x[1])*(pi-x[2])),2)
            qc1.cx(1,2)
            qc1.cx(1,3)
            qc1.u1((2*(pi-x[1])*(pi-y)),3)
            qc1.cx(1,3)
            qc1.cx(2,3)
            qc1.u1((2*(pi-x[2])*(pi-y)),3)
            qc1.cx(2,3)

            
            
            
        if inverse:
            qc1.inverse()
        return qc1
def feature_map():
    return CustomFeatureMap(feature_dimension=3, depth=2)
# the write_and_run function writes the content in this cell into the file "variational_circuit.py"

### WRITE YOUR CODE BETWEEN THESE LINES - START
    
# import libraries that are used in the function below.
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import  RealAmplitudes, EfficientSU2
    
### WRITE YOUR CODE BETWEEN THESE LINES - END
import math

def nCr(n,r):
    f = math.factorial
    return int(f(n) / f(r) / f(n-r))

def variational_circuit():
    # BUILD VARIATIONAL CIRCUIT HERE - START
    
    # import required qiskit libraries if additional libraries are required
    
    # build the variational circuit
    #var_circuit = EfficientSU2(num_qubits=3, su2_gates= ['rx', 'ry'], entanglement='circular', reps=3)
    #var_circuit = EfficientSU2(num_qubits=4, su2_gates= ['rx', 'ry'], entanglement='circular', reps=3)
    
    # BUILD VARIATIONAL CIRCUIT HERE - END
    
    # return the variational circuit which is either a VaritionalForm or QuantumCircuit object
    from qiskit.circuit import QuantumCircuit, ParameterVector

    num_qubits = 4            
    reps = 2              # number of times you'd want to repeat the circuit
    reps_2 = 2

    x = ParameterVector('x', length=reps*(5*num_qubits-1))  # creating a list of Parameters
    qc = QuantumCircuit(num_qubits)

    # defining our parametric form
    for k in range(reps):
            for i in range(num_qubits):
                qc.rx(x[2*i+k*(5*num_qubits-1)],i)
                qc.ry(x[2*i+1+k*(5*num_qubits-1)],i)
            for i in range(num_qubits-1):
                qc.cx(i,i+1)
            for i in range(num_qubits-1):
                qc.rz(2.356194490192345, i)
                qc.rx(1.5707963267948966, i)
                qc.rz(-2.356194490192345, i+1)
                qc.rx(1.5707963267948966, i+1)
                qc.cz(i, i+1)
                qc.rz(-1.5707963267948966, i)
                qc.rx(1.5707963267948966, i)
                qc.rz(x[i+2*(num_qubits)+k*(5*num_qubits-1)], i)
                qc.rx(-1.5707963267948966, i)
                qc.rz(1.5707963267948966, i+1)
                qc.rx(1.5707963267948966, i+1)
                qc.rz(x[i+2*(num_qubits)+k*(5*num_qubits-1)], i+1)
                qc.rx(-1.5707963267948966, i+1)
                qc.cz(i, i+1)
                qc.rz(-1.5707963267948966, i)
                qc.rx(1.5707963267948966, i)
                qc.rz(0.7853981633974483, i)
                qc.rz(-1.5707963267948966, i+1)
                qc.rx(-1.5707963267948966, i+1)
                qc.rz(2.356194490192345, i+1)
            for i in range(num_qubits):
                qc.rx(x[2*i+3*num_qubits-1+k*(5*num_qubits-1)],i)
                qc.ry(x[2*i+1+3*num_qubits-1+k*(5*num_qubits-1)],i)
        
            
    
                
            
    #custom_circ.draw()
    return qc
# # the write_and_run function writes the content in this cell into the file "optimal_params.py"

### WRITE YOUR CODE BETWEEN THESE LINES - START
    
# import libraries that are used in the function below.
import numpy as np
    
### WRITE YOUR CODE BETWEEN THESE LINES - END

def return_optimal_params():
    # STORE THE OPTIMAL PARAMETERS AS AN ARRAY IN THE VARIABLE optimal_parameters 
    
    optimal_parameters =[ 3.63795705 , 1.19128865 ,-0.33132838 , 0.74721243 , 0.92831932 , 0.42240372,
  0.17701117 , 1.80778497,  1.54819545 , 2.14835959 , 0.0773745 , -1.0679474,
  1.10294989 ,-0.69464833,  1.17269839 ,-1.03707315 ,-0.64726175 ,-1.42869361,
 -2.25336805 ,-1.54722281 ,-0.5098562 , -0.83212129 , 0.8265452  , 0.78973385,
 -1.01799251, -0.18191186 , 1.05282282, -0.93642473 ,-0.60919723 ,-2.02198214,
  0.25290384 ,-0.69921034  ,0.52687956, -0.26904783 , 0.01643734 , 0.17502405,
  1.64029289, -0.43707053]
    
    # STORE THE OPTIMAL PARAMETERS AS AN ARRAY IN THE VARIABLE optimal_parameters 
    return np.array(optimal_parameters)
