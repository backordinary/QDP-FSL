# https://github.com/hjaleta/QPE/blob/7dc9862c3cd3404d4dee00cae3a3f91eebd73216/src/QPE/Unitary.py
from types import NoneType
import numpy as np
from QPE.HelpFunctions import float_to_bin, mat_im_to_tuple
from qiskit import QuantumCircuit
from qiskit.quantum_info.operators import Operator
from scipy.stats import unitary_group
from math import pi
from typing import Union, List

class Unitary():
    """
    Class that represents a unitary operator

    ...

    Attributes
    ----------
    operator: qiskit.Operator
        This object describes the unitary matrix/circuit
    
    dim: int
        The qubit-dimension of the Unitary

    eigen: List[complex]
        The eigenvalues of the Unitary
    
    vectors: List[Eigenvector]
        The eigenvectors of the Unitary
    
    phis: List[float]
        The eigenphases of the Unitary
    
    name: str
        The name of the Unitary, decided by the user
    
    Methods
    -------

    get_operator(operator, random, random_state,dim)
        Acquires the appropriate qiskit.Operator object. Either by calculating from input 'operator' or
        by randomizing
    
    gen_random(random_state):
        returns a random qiskit.Operator object
    
    get_dim()
        Returns the qubit-dimension of the Unitary
    
    get_gate(power, ctrl)
        returns the QuantumCircuit representation of the Unitary. 'power' is the exponent of the Unitary,¨
        and 'ctrl' is the number of qubits that control the unitary. 
        'power' = 1 and ctrl = 0 would just return the normal unitary.

    get_eigen()
        calculates the Eigenvector objects associated with the unitary

    get_vectors:
        return a list of lists, every list containing coefficients of an eigenvector in the computational basis
    
    get_phis:
        returns the eigenphases of the unitary
    
    get_phi_bitstrings(m_digits):
        Returns a list of bitstrings corresponding to the eigenphases, all rounded to the closest value
        representable by 'm_digits' bits
    
    get_dict(m_digits):
        Returns a dictionary holding the information about the Unitary


    """
    def __init__(self, operator: Union[QuantumCircuit, Operator, List[list], np.array] = [],
                random:bool = False, random_state:Union[int, NoneType]=None, 
                dim:int = 1, name:str = ""):
        """

        Args:
            operator: QuantumCircuit, Operator, list or numpy.array
                The unitary circuit that should be represented. Can be given on many forms

            random: bool
                If this value is true, a random unitary operator is generated
            
            random_state: int or NoneType
                Fixes the random state seed of the random unitary generation if integer is given.
        
            dim: int
                dimension of the randomized unitary. Defaults to 1.
            name:
                Optional name of the unitary. Will be visible if an experiment is exported to json. Defaults to ""
        """
        self.operator = self.get_operator(operator, random, random_state, dim)
        self.dim = self.get_dim()
        self.eigen = self.get_eigen()
        self.vectors = self.get_vectors()
        self.phis  = self.get_phis()
        self.name = name
    
    def get_operator(self, operator, random, random_state, dim):
        if random == True:
            self.dim = dim
            operator = self.gen_random(random_state)
        else:
            try:
                operator = Operator(operator)
            except:
                raise(ValueError('insert a list, an operator or a circuit'))
        return operator

    def gen_random(self, random_state):
        return Operator(unitary_group.rvs(2**self.dim, random_state=random_state))

    def get_dim(self):
        return int(np.log2(self.operator.dim[0]))

    def get_gate(self, power=1, ctrl=0):  

        changed_op = self.operator.power(power)

        circ_ctr = QuantumCircuit(self.dim)
        circ_ctr.append(changed_op,list(range(self.dim)))
        lab = r'$U^{'+str(power)+r'}$' if power != 1 else 'U'

        if ctrl != 0:
            changed_op = circ_ctr.to_gate(label = lab).control(ctrl)
        else:
            changed_op = circ_ctr.to_gate(label = lab)
    
        return changed_op
    
    def get_eigen(self):
        # self.operator.data
        eigval, eigvect = np.linalg.eig(self.operator.data)
        # print(eigval, eigvect)
        eig = np.concatenate((np.array(eigval).reshape(len(eigval),1), eigvect), axis=1)
        eigenvectors = []
        for i in range(len(eigval)):
            value = eig[i,0]
            vector = eig[i,1:]
            el = Eigenvector(vector, value, i)
            eigenvectors.append(el)

        return eigenvectors
        
    def get_vectors(self):
        vectors = []
        for e in self.eigen:
            vectors.append(e.vector)
        return vectors

    def get_phis(self):
        phis = []
        for e in self.eigen:
            phis.append(e.phi)
        return phis
    
    def get_phi_bitstrings(self, m_digits):
        b_list = []
        for phi in self.phis:
            b = float_to_bin(phi, m_digits)
            b_list.append(b)
        return b_list
    
    def get_dict(self, m_digits):
        d = {}
        d["dimension"] = self.dim
        d["data"] = mat_im_to_tuple(self.operator.data)
        d["eigenvectors"] = mat_im_to_tuple(self.vectors)
        d["phis"] = mat_im_to_tuple(self.phis)
        d["phi_bitstrings"] = self.get_phi_bitstrings(m_digits)
        return d


class Eigenvector():
    """
    This class represents an eigenvector of a Unitary object

    Attributes
    ----------

    vector: numpy.array
        A 1D array containing the complex coefficients of the eigenvector
    
    value: complex
        The eigenvalue corresponding to the vector (defined by the Unitary it belongs to)
    
    index: int
        The index of the eigenvector (defined by the Unitary it belongs to)

    phi: float
        The eigenphase corresponding to the vector (defined by the Unitary it belongs to)

    """
    def __init__(self, vector, value, index):
        self.vector = vector
        self.value = value 
        self.index = index
        self.phi = self.get_phi()    
    
    def get_phi(self):
        phi = np.angle(self.value)/(2*pi)
        if phi < 0:
            phi += 1
        return phi
    
    def get_phi_bitstring(self, m_digits):
        bitstring = float_to_bin(self.phi, m_digits)
        return bitstring


if __name__ == "__main__":
    u = Unitary(random=True)
    d = u.get_dict(3)
    print(d)
