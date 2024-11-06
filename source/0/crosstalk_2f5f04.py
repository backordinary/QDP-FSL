# https://github.com/huafei1137/CrosstalkAdapative/blob/184758193b2cb3a5392426f01ca29b9a529f4c17/crosstalk.py
# Importing standard Qiskit libraries
import sys
from qiskit import QuantumCircuit, QuantumRegister, execute, Aer, IBMQ
from qiskit.compiler import transpile, assemble
from qiskit.tools.jupyter import *
from qiskit.visualization import *
from qiskit.transpiler import PassManager
from qiskit.transpiler import passes
from qiskit.test.mock import FakeMelbourne
from qiskit.transpiler.passes import CrosstalkAdaptiveSchedule
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler import Layout
from z3 import Real, Bool, Sum, Implies, And, Or, Not, Optimize
import logging
from numpy import pi
# with open(sys.argv[1],'r') as f:
#     contents = f.read()
f= sys.argv[1]
print(f)
#quit()
backend = FakeMelbourne()
#print(Aer.backends())
backend2 = Aer.get_backend('statevector_simulator')
#quit()
qreg_q = QuantumRegister(14, 'q')
crosstalk_prop={}
crosstalk_prop[(0,1)]={(2,3):0.9}
crosstalk_prop[(2,3)]={(0,1):0.9}
#crosstalk_prop[(5,6)]={(8,9):0.2}
crosstalk_prop[(8,9)]={(10,11):0.9}
crosstalk_prop[(10,11)]={(8,9):0.9}
circuit = QuantumCircuit(14, 14)
#circuit = QuantumCircuit.from_qasm_file('')
circuit = QuantumCircuit.from_qasm_file(f'/Users/feihua/pythonfile/516projectcode/circuits/large/{f}') 
# circuit.x(qreg_q[0])
# circuit.x(qreg_q[2])
# circuit.h(qreg_q[0])
# circuit.cu1(pi/2, qreg_q[1], qreg_q[0])
# circuit.h(qreg_q[1])
# circuit.cu1(pi/4, qreg_q[2], qreg_q[0])
# circuit.cu1(pi/2, qreg_q[2], qreg_q[1])
# circuit.h(qreg_q[2])
#circuit.cu1(pi/8, qreg_q[3], qreg_q[0])
#circuit.cu1(pi/4, qreg_q[3], qreg_q[1])
#circuit.cu1(pi/2, qreg_q[3], qreg_q[2])
#circuit.h(qreg_q[3])
#circuit.draw(output='text')
mapping = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 ,13]
#layout = Layout({qr[i]: mapping[i] for i in range(14)})
list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10]
#de=circuit.decompose();
circuit.draw(output='mpl',filename='crosstalkqft.jpg')
circuit.qasm(filename='qaoa_10.qasm')

print(circuit.depth())
new_circ = transpile(circuit, backend=backend, 
                     initial_layout=mapping,
                    routing_method='sabre')

#new_circ.draw(output='mpl',filename='crosstalkbase.jpg')
new_circ.qasm(filename='crosstalkqft10.qasm')
print(new_circ.depth())
dag = circuit_to_dag(new_circ)
print(crosstalk_prop)
pass_ = CrosstalkAdaptiveSchedule(backend.properties(), crosstalk_prop)
print(list)
scheduled_dag = pass_.run(dag)
scheduled_circuit = dag_to_circuit(scheduled_dag)
print(scheduled_circuit.depth())
scheduled_circuit.draw(output='mpl',filename='crosstalkqft.jpg')


