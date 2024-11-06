# https://github.com/fatginger1024/PQC_Descriptor/blob/ebc57c542eb5448e8e4c0e1ff07637a4a9e4bc5f/pqc/descriptors/expressibility.py
# -*- coding: utf-8 -*-

import qiskit
import itertools
import typing
import numpy as np
from pqc import Interface
from qiskit.quantum_info import state_fidelity
from scipy.spatial import distance


class Expressibility(Interface):
    
    def __init__(self,circ=None,samples:int=1000,method_ex:str='kl'):
        Interface.__init__(self,circ,samples)
        self._method_ex = method_ex
        #print("Entering expressibility init ...")
        #print("method ex: ",method_ex)


    @staticmethod
    def kl_divergence(prob_a: np.ndarray, prob_b: np.ndarray) -> float:
        """Returns KL divergence between two probabilities"""
        prob_a[prob_a == 0] = 1e-10
        kl_div = np.sum(np.where(prob_a != 0, prob_a * np.log(prob_a / prob_b), 0))
        return typing.cast(float, kl_div)
    
    @staticmethod
    def js_distance(prob_a: np.ndarray, prob_b: np.ndarray,base: float=2.)-> float:
        return distance.jessenshannon(prob_a,prob_b,base)
    
    def prob_haar(self) -> np.ndarray:
        """Returns probability density function of fidelities for Haar Random States"""
        fidelity = np.linspace(0, 1, self._samples)
        num_qubits = self._num_qubits
        
        return (2 ** num_qubits - 1) * (1 - fidelity + 1e-8) ** (2 ** num_qubits - 2)

    
