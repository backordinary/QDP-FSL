# https://github.com/jku-win-se/composition-quantum-circuit/blob/dd95084f9378b5e0fb1f123c346f66c01c532257/examples/qaoa.sample/src-gen/qiskit-qaoa/main-qaoa.py
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
cr=QuantumRegister(4)
QAOA = QuantumCircuit(qr,cr)
#Create objects for quantum operations
e_gate=Elementary_Gates.ElementaryGate()
c_gate=Composite_Gates.CompositeGate()
l_gate=Loop_Operations.LoopOperation()
m_gate=Measurements.MeasurementGate()
target_qubits =[0,1,2,3]
QAOA.append(e_gate.null(target_qubits), target_qubits)
target_qubits =[0,1,2,3]
loop_tqubits=[0,1,2,3]
sl_gate=l_gate.simple_loop(2, [c_gate.cost_unitary_fixed(), c_gate.mixer_unitary(qc.num_qubits)],loopTargetQubits)
qc.append(sl_gate,target_qubits)
target_qubits =[0,1,2,3]
cbits=[0,1,2,3]
QAOA.append(m_gate.measurement(target_qubits,cbits), target_qubits, cbits)
