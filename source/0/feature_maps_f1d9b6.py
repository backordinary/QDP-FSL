# https://github.com/gentinettagian/complexity_of_qsvms/blob/98b8a19a4305d3af5114e1f71935cf93e579f3c7/pegasos_qsvm/feature_maps.py
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from model_base import Model

class _FeatureMap(Model):
    """Base class for feature maps encoding classical data in quantum circuits."""
    name = 'FeatureMap'

    def __init__(self, q, r, mapping='loop'):
        """
        :param q: number of qubits
        :param r: number of repetitions in the feature map
        :param circuit: QuantumCircuit, representing the feature map with free parameters
        :param mapping: str, one of 'vertical', 'shifted' or 'loop'. Determines how the classical input vector is mapped
                to the rotation parameters in the fixed feature map circuit architecture.
        """
        self.q = q
        self.r = r
        self.circuit = QuantumCircuit(q)
        self.mapping = mapping

        self.designator_loose = f'fm  = {self.name}\nr   = {self.r}\nmap = {self.mapping}'
        self.designator = self.designator_loose.replace(' ', '').replace('\n', ',')

    def get_param_dict(self, x):
        """Get a dictionary associating the free parameters in the circuit with a vector of classical input.
        :param x: ndarray, shape (s,) containing a single input datum
        :return: dict, containing qiskit Parameter objects as the keys and the components of the datum as the values 
        """
        # size of classical input
        s = x.shape[0]
        
        fm_param_dict = {}

        if self.mapping == 'vertical':
            for i in range(self.r):
                for j in range(self.q):
                    fm_param_dict[self.circuit.parameters[i*self.q + j]] = x[j % s]

        elif self.mapping == 'shifted':
            for i in range(self.r):
                for j in range(self.q):
                    fm_param_dict[self.circuit.parameters[i*self.q + j]] = x[(j+i) % s]

        elif self.mapping == 'loop':
            for i in range(self.circuit.num_parameters):
                fm_param_dict[self.circuit.parameters[i]] = x[i % s]

        else:
            raise ValueError(f'Unsopported mapping: {self.mapping}')

        return fm_param_dict

    def get_bound_circuit(self, x):
        """Get a QuantumCircuit where the free parameters in self.circuit are bound to values given by an input datum.
        :param x: ndarray, shape (s,) containing a single input datum
        :return: QuantumCircuit, with no free parameters
        """
        param_dict = self.get_param_dict(x)

        return self.circuit.bind_parameters(param_dict)

    def get_reduced_params_circuit(self):
        """Reduces the number of free parameters in the circuit to q. This means that subsequent layers contain the same 
        parameters. The behavior is then like in ZFeatureMap and ZZFeaturemap. This is for example needed when using 
        QSVM and QuantumKernel from qiskit_machine_learning.
        :return: QuantumCircuit, with q free parameters repeating in the layers
        """
        new_params = ParameterVector('ϕ', self.q)

        fm_param_dict = {}
        for i, param in enumerate(self.circuit.parameters):
            fm_param_dict[param] = new_params[i % len(new_params)]

        return self.circuit.assign_parameters(fm_param_dict, inplace=False)



class EasyFeatureMap(_FeatureMap):
    """Feature map with no entanglement."""
    name = 'easy'

    def __init__(self, q, r, mapping='loop', hadamard=True):
        """
        :param q: number of qubits
        :param r: number of repetitions in the feature map
        :param circuit: QuantumCircuit, representing the feature map with free parameters
        :param hadamard: bool, whether to use Hadamard gates at the start of the layers or not
        """
        super().__init__(q, r, mapping)

        phis = ParameterVector('φ', q*r)
        
        # loop over r repetitions
        for i in range(r):
            if hadamard:
                # Hadamard gates
                for j in range(q):
                    self.circuit.h(j)

            # rz rotations
            for j in range(q):
                self.circuit.rz(np.pi*phis[i*q + j], j)

            # self.circuit.barrier()
        
class MediumFeatureMap(_FeatureMap):
    """Feature map with nearest neighbor entanglement."""
    name = 'medi'

    def __init__(self, q, r, mapping='loop', hadamard=True):
        """
        :param q: number of qubits
        :param r: number of repetitions in the feature map
        :param circuit: QuantumCircuit, representing the feature map with free parameters
        :param hadamard: bool, whether to use Hadamard gates at the start of the layers or not
        """
        super().__init__(q, r, mapping)

        phis = ParameterVector('φ', q*r)
        
        # loop over r repetitions
        for i in range(r):
            if hadamard:
                # Hadamard gates
                for j in range(q):
                    self.circuit.h(j)

            # rz rotations
            for j in range(q):
                self.circuit.rz(np.pi*phis[i*q + j], j)

            # nearest neighbor RZZ
            for j in range(q):
                if j < q-1:
                    self.circuit.rzz(np.pi*phis[i*q + j]*phis[(i*q+1) + j], j, j+1)

                    # equivalent to
                    # self.circuit.cx(j, j+1)
                    # self.circuit.rz(np.pi*phis[i*q + j]*phis[(i*q+1) + j], j+1)
                    # self.circuit.cx(j, j+1)

            # self.circuit.barrier()
        
class HardFeatureMap(_FeatureMap):
    """Feature map with all qubit entanglement."""
    name = 'hard'

    def __init__(self, q, r, mapping='loop', hadamard=True):
        """
        :param q: number of qubits
        :param r: number of repetitions in the feature map
        :param circuit: QuantumCircuit, representing the feature map with free parameters
        :param hadamard: bool, whether to use Hadamard gates at the start of the layers or not
        """
        super().__init__(q, r, mapping)

        phis = ParameterVector('φ', q*r)
        
        # loop over r repetitions
        for i in range(r):
            if hadamard:
                # Hadamard gates
                for j in range(q):
                    self.circuit.h(j)
            
            # rz rotations
            for j in range(q):
                self.circuit.rz(np.pi*phis[i*q + j], j)

            # CNOT between all qubits
            for j in range(1,q):
                for k in range(j):
                    if j < q and k < q-1:
                        self.circuit.cx(k, j)

                # self.circuit.barrier()
        
class HadamardFeatureMap(_FeatureMap):
    """Feature map solely consisting of Hadamard gates with no input data dependence."""
    name = 'hada'

    def __init__(self, q, r):
        """
        :param q: number of qubits
        :param r: number of repetitions in the feature map
        :param circuit: QuantumCircuit, representing the feature map with free parameters
        """
        super().__init__(q, r, mapping='vertical')

        # initial Hadamard gates
        for i in range(q):
            self.circuit.h(i)
        # self.circuit.barrier()
        
    def get_param_dict(self, x):
        """Get a dictionary associating the free parameters in the circuit with a vector of classical input.
        :param x: ndarray, shape (s,) containing a single input datum
        :return: dict, empty 
        """

        return {}

    def bind_parameters(self, x):
        """Get a QuantumCircuit where the free parameters in self.circuit are bound to values given by an input datum.
        :param x: ndarray, shape (s,) containing a single input datum
        :return: QuantumCircuit, with no free parameters
        """

        return self.circuit

# legacy

class EasyFeatureMap_legacy(_FeatureMap):
    """Feature map with no entanglement. This version has the original coefficients in the rotations and
    always used before 04.06.2021"""
    name = 'easy'

    def __init__(self, q, r, mapping='loop', hadamard=True):
        """
        :param q: number of qubits
        :param r: number of repetitions in the feature map
        :param circuit: QuantumCircuit, representing the feature map with free parameters
        :param hadamard: bool, whether to use Hadamard gates at the start of the layers or not
        """
        super().__init__(q, r, mapping)

        phis = ParameterVector('φ', q*r)
        
        # loop over r repetitions
        for i in range(r):
            if hadamard:
                # Hadamard gates
                for j in range(q):
                    self.circuit.h(j)

            # rz rotations
            for j in range(q):
                self.circuit.rz(2*np.pi*phis[i*q + j], j)

            # self.circuit.barrier()

class MediumFeatureMap_legacy(_FeatureMap):
    """Feature map with nearest neighbor entanglement. This version has the original coefficients in the rotations and
    always used before 04.06.2021"""
    name = 'medi_legacy'

    def __init__(self, q, r, mapping='loop', hadamard=True):
        """
        :param q: number of qubits
        :param r: number of repetitions in the feature map
        :param circuit: QuantumCircuit, representing the feature map with free parameters
        :param hadamard: bool, whether to use Hadamard gates at the start of the layers or not
        """
        super().__init__(q, r, mapping)

        phis = ParameterVector('φ', q*r)
        
        # loop over r repetitions
        for i in range(r):
            if hadamard:
                # Hadamard gates
                for j in range(q):
                    self.circuit.h(j)

            # rz rotations
            for j in range(q):
                self.circuit.rz(2*np.pi*phis[i*q + j], j)

            # nearest neighbor RZZ
            for j in range(q):
                if j < q-1:
                    self.circuit.rzz((2*np.pi)**2*phis[i*q + j]*phis[(i*q+1) + j], j, j+1)
            
            # self.circuit.barrier()
