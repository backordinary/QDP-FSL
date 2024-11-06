# https://github.com/jku-win-se/composition-quantum-circuit/blob/dd95084f9378b5e0fb1f123c346f66c01c532257/examples/quantum.counting.sample/src-gen/qiskit-quantum-counting-alt-1/Elementary_Gates.py
#elementary Quantum Gates
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit import Gate
import numpy as np
from numpy.random import choice, random as rand, randint

class ElementaryGate:
    def __init__(self):
        self.type="ElementaryGate"

    def rx_gate(self,target_qubits, control_qubits=None, beta=None, inverse=False):
        if beta==None:
            beta=2*np.pi*rand(1)[0]
        if control_qubits==None:
            control_qubits=[]
        qc=QuantumCircuit(len(target_qubits))
        for i in range(len(target_qubits)):
            qc.rx(beta, i)
        gate = qc.to_gate()
        if control_qubits!=None:
            for j in range(len(control_qubits)):
                gate=gate.control()
        if inverse==True:
            gate=gate.inverse()
        return gate

    def hadamard(self,target_qubits, control_qubits=None,inverse=False):
        if control_qubits==None:
            control_qubits=[]
        qc=QuantumCircuit(len(target_qubits))
        for i in range(len(target_qubits)):
            qc.h(i)
        gate = qc.to_gate()
        if control_qubits!=None:
            for j in range(len(control_qubits)):
                gate=gate.control()
        if inverse==True:
            gate=gate.inverse()
        return gate


    def swap(self,target_qubits=None, control_qubits=None, inverse=False):
        qc=QuantumCircuit(2)
        qc.swap(0,1)
        swap=qc.to_gate()
        return swap

    def p_gate(self,theta, target_qubits, control_qubits=None, inverse=False):
        if control_qubits==None:
            control_qubits=[]
        qc=QuantumCircuit(len(target_qubits))
        for i in range(len(target_qubits)):
            qc.p(theta, i)
        gate = qc.to_gate()
        if inverse==True:
            gate=gate.inverse()
        if control_qubits!=None:
            for j in range(len(control_qubits)):
                gate=gate.control()
        return gate
