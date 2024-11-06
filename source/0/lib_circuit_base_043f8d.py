# https://github.com/weiwenjiang/qc_mnist/blob/4e9fa66d9fc30c655b640009d36d7773b4053855/VQC/libs/circuit/lib_circuit_base.py

import sys
import numpy as np
import numpy as np
from qiskit.tools.monitor import job_monitor
from qiskit import QuantumRegister
from qiskit.extensions import  UnitaryGate
from qiskit import Aer, execute,IBMQ,transpile
import math
from qiskit import BasicAer
import copy
import abc

class BaseCircuit(metaclass= abc.ABCMeta):
    def __init__(self,n_qubits,n_repeats):
        """
      param n_qubits: input qubits of each unit
      param n_repeats: repeat times of each unit
        """    
        self.n_qubits = n_qubits
        self.n_repeats = n_repeats
    
    def add_qubits(self,circuit,name,number):
        qubits = QuantumRegister(number,name)
        circuit.add_register(qubits)
        return qubits
    
    def add_input_qubits(self,circuit,name):
        inps = []
        for i in range(self.n_repeats):
            inp = QuantumRegister(self.n_qubits,name+str(i)+"_qbit")
            circuit.add_register(inp)
            inps.append(inp)
        return inps
    
    @abc.abstractclassmethod
    def forward(self,circuit):
        pass


    