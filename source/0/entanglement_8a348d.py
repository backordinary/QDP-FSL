# https://github.com/fatginger1024/PQC_Descriptor/blob/ebc57c542eb5448e8e4c0e1ff07637a4a9e4bc5f/pqc/descriptors/entanglement.py
# -*- coding: utf-8 -*-

import qiskit
import itertools
import numpy as np
from pqc import Interface
from qiskit.quantum_info import partial_trace

class Entanglement_Capability(Interface):
    
    def __init__(self,circ=None,samples:int=1000,method_ec:str='mw'):
        Interface.__init__(self,circ,samples)
        self._method_ec = method_ec
        #print("Entering entanglement init ...")
        #print("method ec: ",method_ec)

    @staticmethod
    def scott_helper(state, perms):
        
        dems = np.linalg.matrix_power(
            [partial_trace(state, list(qb)).data for qb in perms], 2
        )
        trace = np.trace(dems, axis1=1, axis2=2)
        return np.sum(trace).real

    def meyer_wallach_measure(self,states, num_qubits):
        r"""Returns the meyer-wallach entanglement measure for the given circuit.
        .. math::
            Q = \frac{2}{|\vec{\theta}|}\sum_{\theta_{i}\in \vec{\theta}}
            \Bigg(1-\frac{1}{n}\sum_{k=1}^{n}Tr(\rho_{k}^{2}(\theta_{i}))\Bigg)
        """
        num_qubits = self._num_qubits
        permutations = list(itertools.combinations(range(num_qubits), num_qubits - 1))
        ns = 2 * sum(
            [
                1 - 1 / num_qubits * self.scott_helper(state, permutations)
                for state in states
            ]
        )
        
        return ns.real
 
    def scott_measure(self,states, num_qubits):
        r"""Returns the scott entanglement measure for the given circuit.
        .. math::
            Q_{m} = \frac{2^{m}}{(2^{m}-1) |\vec{\theta}|}\sum_{\theta_i \in \vec{\theta}}\
            \bigg(1 - \frac{m! (n-m)!)}{n!}\sum_{|S|=m} \text{Tr} (\rho_{S}^2 (\theta_i)) \bigg)\
            \quad m= 1, \ldots, \lfloor n/2 \rfloor
        """
        num_qubits = self._num_qubits
        m = range(1, num_qubits // 2 + 1)
        permutations = [
            list(itertools.combinations(range(num_qubits), num_qubits - idx))
            for idx in m
        ]
        combinations = [1 / comb(num_qubits, idx) for idx in m]
        contributions = [2 ** idx / (2 ** idx - 1) for idx in m]
        ns = []

        for ind, perm in enumerate(permutations):
            ns.append(
                contributions[ind]
                * sum(
                    [
                        1 - combinations[ind] * self.scott_helper(state, perm)
                        for state in states
                    ]
                )
            )

        return np.array(ns)
    
  
    


