# https://github.com/jku-win-se/composition-quantum-circuit/blob/dd95084f9378b5e0fb1f123c346f66c01c532257/examples/quantum.counting.sample/src-gen/qiskit-quantum-counting-alt-2/main-quantum-counting-alt-2.py
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit import Gate
import numpy as np
#Import Library
import Composite_Gates
import Loop_Operations
import Measurements
import Elementary_Gates
#Create empty Quantum Circuit
qr=QuantumRegister(4)
qr1=QuantumRegister(4)
cr=QuantumRegister(3)
QuantumCounting = QuantumCircuit(qr,qr1,cr)
#Create objects for quantum operations
e_gate=Elementary_Gates.ElementaryGate()
c_gate=Composite_Gates.CompositeGate()
l_gate=Loop_Operations.LoopOperation()
m_gate=Measurements.MeasurementGate()
target_qubits =[0,1,2,3,4,5,6,7]
QuantumCounting.append(e_gate.null(target_qubits), target_qubits)
target_qubits =[0,1,2,3,0,1,2,3]
loop_tqubits=[0,1,2,3]
loop_cqubits=[0,1,2,3]
QuantumCounting.append(l_gate.Power_2_loop(c_gate.grover4,loop_target_qubits,loop_control_qubits, increment_t=True), target_qubits)
target_qubits =[0,1,2,3]
target_qubits =[0,1,2,3]
cbits=[0,1,2,3]
QuantumCounting.append(m_gate.measurement(target_qubits,cbits), target_qubits, cbits)
