# https://github.com/vietphamngoc/QPAC/blob/e64d34d10d3e9dca993717707c5da5e8d9d70a93/oracle.py
import numpy as np
import random

from qiskit import QuantumCircuit

class Oracle:

    def __init__(self, n: int, logic: list, params: list=[]):
        """
        Instanciates an object of the Oracle class which is the query oracle
        for the target function.

        Arguments:
            - n: int, the dimension of the input space
            - logic: list, the list of the controlled X gates corresponding to the target function
            - params: list (default=[]), the list of the angles applied to the roation gates to create the distribution. If empty will create a random one

        Returns:
            - An object of the class Oracle with attributes:
                * dim: the dimension of the input space
                * params: the list of the angles applied to the roation gates to create the distribution
                * logic: the list of the controlled X gates corresponding to the target function
                * gate: the quantum gate corresponding to the oracle
                * inv_gate: the inverse of gate
        """
        if params == []:
            for i in range(n):
                params.append(random.random()*np.pi)

        if len(params) != n:
            raise ValueError(f"The length of param is {len(params)}, it should be {n}")

        self.dim = n
        self.params = params
        self.logic = self.get_logic(logic)

        qc = QuantumCircuit(n+1)
        self.__apply_amplitude(qc, inverse=False)
        self.__apply_function(qc)

        qc_inv = QuantumCircuit(n+1)
        self.__apply_function(qc_inv)
        self.__apply_amplitude(qc_inv, inverse=True)

        self.gate = qc.to_gate(label="Oracle")
        self.inv_gate = qc_inv.to_gate(label="Oracle^-1")


    def get_logic(self, logic):
        return(logic)


    def __apply_amplitude(self, qc, inverse):
        for i in range(self.dim):
            qc.ry((-1)**(inverse)*self.params[i],i)


    def __apply_function(self, qc):
        for u in self.logic:
            if len(u) != self.dim:
                raise ValueError(f"The length of {u} should be {self.dim}")
            if u == "0"*self.dim:
                qc.x(self.dim)
            else:
                controls = []
                for i in range(self.dim):
                    if u[i] == "1":
                        controls.append(i)
                qc.mcx(controls, self.dim)



class Parity_Oracle(Oracle):

    def get_logic(self, logic):
        if len(logic) != 1:
            raise ValueError("For parity function, one input must be provided")
        log = []
        for i in range(self.dim):
            if logic[0][i] == "1":
                u = ""
                for j in range(self.dim):
                    if i == j:
                        u += "1"
                    else:
                        u += "0"
                log.append(u)
        return(log)

