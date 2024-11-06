# https://github.com/oimichiu/quantumGateModel/blob/fa1cb5ed751edbebe512ee299d5484949c856340/IBMQX/qiskit-tutorials/coduriCareNUcompileaza/tutorial_programming_IBM/localSimulator/1/compilation.py
import sys
import os

# Checking the version of PYTHO; we only support > 3.5
if sys.version_info < (3,5):
    raise Exception('Please use Python version 3.5 or greater.')

# import qiskit from qiskit-sdk-py folder 
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..', 'qiskit-sdk-py'))
from qiskit import QuantumProgram
import Qconfig

import math
from pprint import pprint

qp = QuantumProgram()
# quantum register for the first circuit
q1 = qp.create_quantum_register('q1', 4)
c1 = qp.create_classical_register('c1', 4)

# quantum register for the second circuit
q2 = qp.create_quantum_register('q2', 2)
c2 = qp.create_classical_register('c2', 2)

# making the first circuits
qc1 = qp.create_circuit('GHZ', [q1], [c1])
qc2 = qp.create_circuit('superposition', [q2], [c2])
qc1.h(q1[0])
qc1.cx(q1[0], q1[1])
qc1.cx(q1[1], q1[2])
qc1.cx(q1[2], q1[3])
qc1.measure(q1[0], c1[0])
qc1.measure(q1[1], c1[1])
qc1.measure(q1[2], c1[2])
qc1.measure(q1[3], c1[3])

# making the second circuits
qc2.h(q2)
qc2.measure(q2[0], c2[0])
qc2.measure(q2[1], c2[1])

# printing the circuits
# print(qp.get_qasm('GHZ'))
# print(qp.get_qasm('superposition'))

# converting to qobj
qobj = qp.compile(['GHZ', 'superposition'], backend = 'local_qasm_simulator')
# qp.get_execution_list(qobj)

# get the configuration for one circuit
# qp.get_compiled_configuration(qobj, 'GHZ')

# get the compiled qasm used
print(qp.get_compiled_qasm(qobj, 'GHZ'))