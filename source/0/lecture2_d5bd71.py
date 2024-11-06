# https://github.com/oimichiu/quantumGateModel/blob/fa1cb5ed751edbebe512ee299d5484949c856340/IBMQX/qiskit-tutorials/coduriCareNUcompileaza/tutorial_programming_IBM/youtube/lecture2.py
import sys
import os
from pprint import pprint
# so we need a relative position from this file path.
# TODO: Relative imports for intra-package imports are highly discouraged.
# http://stackoverflow.com/a/7506006
sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'qiskit-sdk-py'))
from qiskit import QuantumProgram

qp = QuantumProgram()
qr = qp.create_quantum_register('qr', 2)
cr = qp.create_classical_register('cr', 2)
qc = qp.create_circuit('qc', [qr],[cr])

circuit = qp.get_circuit('qc')
quantum_r = qp.get_quantum_register('qr')
classical_r = qp.get_classical_register('cr')

circuit.x(quantum_r[0])
circuit.y(quantum_r[1])

circuit.cx(quantum_r[0], quantum_r[1])

circuit.measure(quantum_r[0], classical_r[0])
circuit.measure(quantum_r[1], classical_r[1])

backend = 'local_qasm_simulator'
circuits = ['qc']

result = qp.execute(circuits, backend, wait=2, timeout=240)
print(result)
result.get_counts('qc')
out = result.get_ran_qasm('qc')
print(out)