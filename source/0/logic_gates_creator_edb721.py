# https://github.com/benkoehlL/Qiskit_Playground/blob/b52f7cf1d0519f71ab1f6c1e79930d280e2a1f80/logic_gates_creator.py
'''
This program creates a .tex file of several classical logic gates
'''

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
import os
import shutil

logic_folder = os.getcwd()
LaTex_folder_logic_gates = str(os.getcwd())+'/Latex_quantum_gates/logic_gates/'
if not os.path.exists(LaTex_folder_logic_gates):
    os.makedirs(LaTex_folder_logic_gates)
else:
    shutil.rmtree(LaTex_folder_logic_gates)
    os.makedirs(LaTex_folder_logic_gates)
print(LaTex_folder_logic_gates)
## NOT gate
q = QuantumRegister(1, name='q')
c = ClassicalRegister(1, name='c')
qc = QuantumCircuit(q,c)
qc.x(q[0])
qc.measure(q[0],c[0])
LaTex_code = qc.draw(output='latex_source', justify=None) # draw the circuit
f_name = 'NOT_gate.tex'
with open(LaTex_folder_logic_gates+f_name, 'w') as f:
            f.write(LaTex_code)
## AND gate
q = QuantumRegister(3, name='q')
qubit_state = '0'
c = ClassicalRegister(1, name='c')
qc = QuantumCircuit(q,c)
qc.reset(q[2])
qc.ccx(q[0], q[1], q[2])
qc.measure(q[2], c[0])
LaTex_code = qc.draw(output='latex_source', justify=None) # draw the circuit
f_name = 'AND_gate.tex'
with open(LaTex_folder_logic_gates+f_name, 'w') as f:
            f.write(LaTex_code)

## OR gate
q = QuantumRegister(3, name='q')
c = ClassicalRegister(1, name='c')
qc = QuantumCircuit(q,c)
qc.reset(q[2])
qc.ccx(q[0], q[1], q[2])
qc.cx(q[0],q[2])
qc.cx(q[1],q[2])
qc.measure(q[2],c[0])
LaTex_code = qc.draw(output='latex_source', justify=None) # draw the circuit
f_name = 'OR_gate.tex'
with open(LaTex_folder_logic_gates+f_name, 'w') as f:
            f.write(LaTex_code)

## XOR gate
q = QuantumRegister(3, name='q')
c = ClassicalRegister(1, name='c')
qc = QuantumCircuit(q,c)
qc.reset(q[2])
qc.cx(q[0],q[2])
qc.cx(q[1],q[2])
qc.measure(q[2],c[0])
LaTex_code = qc.draw(output='latex_source', justify=None) # draw the circuit
f_name = 'XOR_gate.tex'
with open(LaTex_folder_logic_gates+f_name, 'w') as f:
            f.write(LaTex_code)

## NOR gate
q = QuantumRegister(3, name='q')
c = ClassicalRegister(1, name='c')
qc = QuantumCircuit(q,c)
qc.reset(q[2])
qc.ccx(q[0], q[1], q[2])
qc.cx(q[0],q[2])
qc.cx(q[1],q[2])
qc.x(q[2])
qc.measure(q[2],c[0])
LaTex_code = qc.draw(output='latex_source', justify=None) # draw the circuit
f_name = 'NOR_gate.tex'
with open(LaTex_folder_logic_gates+f_name, 'w') as f:
            f.write(LaTex_code)