# https://github.com/jku-win-se/composition-quantum-circuit/blob/dd95084f9378b5e0fb1f123c346f66c01c532257/examples/quantum.counting.sample/src-gen/qiskit-quantum-counting-alt-2/Composite_Gates.py
#composite Gates
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit import Gate
import numpy as np
from numpy.random import choice, random as rand, randint

import Composite_Gates
import Measurements
import Elementary_Gates
import Loop_Operations

class CompositeGate:
    def __init__(self):
        self.type="CompositeGate"

    def cost_unitary_fixed(self):
        w=[[1,2,3,4], [6,2,5], [5,3], [4]]
        qc=QuantumCircuit(4)
        gamma=np.pi*rand(1)[0]
        for j in range(0+1,4):
            qc.cnot(0,j)
            qc.rz(2*gamma*w[0][j-0],j)
            qc.cnot(0,j)
        for j in range(1+1,4):
            qc.cnot(1,j)
            qc.rz(2*gamma*w[1][j-1],j)
            qc.cnot(1,j)
        for j in range(2+1,4):
            qc.cnot(2,j)
            qc.rz(2*gamma*w[2][j-2],j)
            qc.cnot(2,j)
        for j in range(3+1,4):
            qc.cnot(3,j)
            qc.rz(2*gamma*w[3][j-3],j)
            qc.cnot(3,j)
        #diagonal elements of matrix
        qc.barrier()
        for i in range(len(w)):
            qc.rz(2*gamma*w[i][0],i)
        c_u=qc.to_instruction()
        return c_u

    def mixer_unitary(self,size):
        qc=QuantumCircuit(size)
        beta=np.pi*rand(1)[0]
        for i in range(size):
            qc.rx(2*beta,i)
        m_u=qc.to_instruction()
        return m_u

    def qaoa_unitary(self,num_qubits):
        #block from Circuit Generation (see QAOA_main)
        qc1=QuantumCircuit(num_qubits)
        c_gate=Composite_Gates.CompositeGate()
        l_gate=Loop_Operations.LoopOperation()
        e_gate=Elementary_Gates.ElementaryGate()
        m_gate=Measurements.MeasurementGate()
        m_u=c_gate.mixer_unitary(num_qubits)
        c_u=c_gate.cost_unitary_fixed()
        target_qubits=[0,1,2,3] #for Cost Unitary
        qc1.append(c_u,target_qubits)
        target_qubits=[0,1,2,3] #for Mixer
        qc1.append(m_u,target_qubits)
        qaoa_unitary=qc1.to_instruction()

        #additionally
        return qaoa_unitary

    def grover3(self,control_qubits=None, inverse=False):
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.h(1)
        qc.h(2)
        #oracle --> specifically for this use case
        qc.cz(0, 2)
        qc.cz(1, 2)
        qc.h(0)
        qc.h(1)
        qc.h(2)
        qc.x(0)
        qc.x(1)
        qc.x(2)
        qc.h(2)
        qc.mct(list(range(2)), 2)
        qc.h(2)
        qc.x(0)
        qc.x(1)
        qc.x(2)
        qc.h(0)
        qc.h(1)
        qc.h(2)
        grover = qc.to_gate()
        grover.name = "Grover"
        if inverse==True:
            grover=grover.inverse()
        if control_qubits!=None:
            for i in range(len(control_qubits)):
                grover=grover.control()
        return grover

    def grover4(self,control_qubits=None, inverse=False):
        qc = QuantumCircuit(4)
        # Oracle
        qc.h([2,3])
        qc.ccx(0,1,2)
        qc.h(2)
        qc.x(2)
        qc.ccx(0,2,3)
        qc.x(2)
        qc.h(3)
        qc.x([1,3])
        qc.h(2)
        qc.mct([0,1,3],2)
        qc.x([1,3])
        qc.h(2)
        # Diffuser
        qc.h(range(3))
        qc.x(range(3))
        qc.z(3)
        qc.mct([0,1,2],3)
        qc.x(range(3))
        qc.h(range(3))
        qc.z(3)
        grover=qc.to_gate()
        if inverse==True:
            grover=grover.inverse()
        if control_qubits!=None:
            for i in range(len(control_qubits)):
                grover=grover.control()
        return grover

    def qft(self,qubits, inverse=False):
        n = len(qubits)
        #Creates an n-qubit QFT_dagger circuit
        circuit = QuantumCircuit(n)
        def swap_registers(circuit, n):
           for qubit in range(n//2):
               circuit.swap(qubit, n-qubit-1)
           return circuit
        def qft_rotations(circuit, n):
            #Performs qft on the first n qubits in circuit (without swaps)
            if n == 0:
                return circuit
            n -= 1
            circuit.h(n)
            for qubit in range(n):
                circuit.cp(np.pi/2**(n-qubit), qubit, n)
            qft_rotations(circuit, n)

        qft_rotations(circuit, n)
        swap_registers(circuit, n)
        qft_gate = circuit.to_gate()
        if inverse==True:
            qft_gate=qft_gate.inverse()
        return qft_gate

    def qft_elements(self,target_qubits, control_qubits=None, inverse=False):
        qc=QuantumCircuit(len(target_qubits))
        n=len(target_qubits)
        for i in range(n-1):
            qc.cp(-np.pi/(2**(i+1)), n-2-i, n-1)
        qc.h(n-1)
        gate=qc.to_gate()
        if inverse==True:
            gate=gate.inverse()
        if control_qubits!=None:
            for i in range(len(control_qubits)):
                gate=gate.control()
        return gate

    def qft_elements2(self,target_qubits, control_qubits=None, inverse=False):
        qubits=len(target_qubits)
        if qubits==1:
            qc=QuantumCircuit(qubits)
            qc.h(qubits-1)
            gate=qc.to_instruction()
            return gate
        else:
            theta=-np.pi/2
            p_target=[1]
            p_control=[0]
            target_qubits=[qubits-1]
            control_qubits=[]
            for i in range(qubits-1):
                control_qubits.append(i)
            qc=QuantumCircuit(qubits)
            qc.append(Loop_Operations.LoopOperation().Power_2_loop(Elementary_Gates.ElementaryGate().p_gate(theta, p_control, p_target),target_qubits,control_qubits, increment_t=True, increment_c=False), qc.qubits)
            qc.h(qubits-1)
            gate=qc.to_instruction()
            if inverse==True:
                gate=gate.inverse()
            return gate
