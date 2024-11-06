# https://github.com/sqrta/Qsketch/blob/ffdfbe309e6d18f562d4e2cab968c45cfee68ba1/Qiskit_wstate.py
from qiskit import QuantumCircuit
from qiskit import gate
from qiskit.circuit.library.standard_gates import HGate

circ = QuantumCircuit(3)

# Spec { 'qubit':2, 'initial': |00> } 
circ.__1(__, 0)
ch0=HGate().control(ctrl_sstate=0)
circ.append(ch0, [0,1])
# Spec { final: ¹/√3 (|00⟩ + |01⟩ + |10⟩) }
