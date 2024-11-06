# https://github.com/CodieKev/Variational_Quantum_Classifier_CKT/blob/a10dab69a2434ed6c9bd701fec6c13311b6d7b9f/Custon_5_Feature_Mapping/feature_map.py
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
        self._num_qubits = self._feature_dimension = feature_dimension+2
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
        qc = QuantumCircuit(5)

        
        for _ in range(self._depth):
            y = -1.3*x[0]+x[1]
            z = (0.130554 + 0.087421 *(x[0]**2) + 0.193981* (x[1]**2) + x[0] *(0.14809 - 0.248248 *x[1] - 0.0651 *x[2]) +x[1]*(-0.140875 - 0.0429*x[2]) - 0.319747*x[2] + 0.270152 *(x[2]**2))*13
            qc.h(0)
            qc.h(1)
            qc.h(2)
            qc.h(3)
            qc.u1(x[0],0)
            qc.u1(x[1],1)
            qc.u1(x[2],2)
            qc.u1(y,3)
            qc.u1(z,3)
            qc.cx(0,1)
            qc.u1((2*(pi-x[0])*(pi-x[1])),1)
            qc.cx(0,1)
            qc.cx(0,2)
            qc.u1((2*(pi-x[0])*(pi-x[2])),2)
            qc.cx(0,2)
            qc.cx(0,3)
            qc.u1((2*(pi-x[0])*(pi-y)),3)
            qc.cx(0,3)
            qc.cx(1,2)
            qc.u1((2*(pi-x[1])*(pi-x[2])),2)
            qc.cx(1,2)
            qc.cx(1,3)
            qc.u1((2*(pi-x[1])*(pi-y)),3)
            qc.cx(1,3)
            qc.cx(2,3)
            qc.u1((2*(pi-x[2])*(pi-y)),3)
            qc.cx(2,3)
            qc.cx(0,4)
            qc.u1((2*(pi-x[0])*(pi-z)),4)
            qc.cx(0,4)
            qc.cx(1,4)
            qc.u1((2*(pi-x[1])*(pi-z)),4)
            qc.cx(1,4)
            qc.cx(2,4)
            qc.u1((2*(pi-x[2])*(pi-z)),4)
            qc.cx(2,4)
            qc.cx(3,4)
            qc.u1((2*(pi-y)*(pi-z)),4)
            qc.cx(3,4)
            
            
            
            
        if inverse:
            qc.inverse()
        return qc
def feature_map():
    return CustomFeatureMap(feature_dimension=3, depth=2)
