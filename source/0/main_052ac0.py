# https://github.com/harukimisola/demo/blob/691c8bf824acd6401f88478e6d428a1aef3e7464/main.py
import numpy as np
import math
from qiskit import QuantumCircuit, transpile, assemble, Aer, IBMQ, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RYGate
from qiskit.providers.aer.library import save_statevector



pro = np.array([1,2,3,4,5,6,7,8])
pro = pro / np.linalg.norm(pro)


def first(n):
  a = np.arange(1, 2**(n), 2)
  a = a.tolist()
  a[1:-1] = list(reversed(a[1:-1]))
  a = [i-1 for i in a ]
  return a

def second(n):
  a = np.arange(2, 2 ** (n)+1, 2)
  a = a.tolist()
  a[1:-1] = list(reversed(a[1:-1]))
  a = [i-1 for i in a]
  return a


num = first(3)    #qubit个数
a = pro[num]
cosx1 = np.linalg.norm(a)
x1 = math.acos(cosx1) * 2

a_ = []
for i in range(0,len(a),2):
  ax = a[i:i+2]
  a_.append(ax)

cosx2 = np.linalg.norm(a_[0]) / cosx1
x2 = math.acos(cosx2) * 2
cosx4 = a_[0][0] / (cosx2*cosx1)
x4 = math.acos(cosx4)  * 2
cosx5 = a_[1][0] / (math.sin(x2/2)*cosx1)

num_2 = second(3)
b = pro[num_2]
b_ = []
for i in range(0,len(b),2):
  bx = b[i:i+2]
  b_.append(bx)

cosx3 = np.linalg.norm(b_[0]) / math.sin(x1/2)
x3 = math.acos(cosx3) * 2
cosx6 = b_[0][0] / (math.cos(x3 /2) * math.sin(x1 /2 ))
cosx7 = b_[1][0] / (math.sin(x3/2) * math.sin(x1 / 2))
x7 = math.acos(cosx7) * 2
x5 = math.acos(cosx5) *2
x6 = math.acos(cosx6)*2

print(x1,x2,x3,x4,x5,x6,x7)

rry4 = RYGate(x4).control(num_ctrl_qubits=1,ctrl_state=0).control(num_ctrl_qubits=1,ctrl_state=0)
rry5 = RYGate(x5).control(num_ctrl_qubits=1,ctrl_state=1).control(num_ctrl_qubits=1,ctrl_state=0)
rry6 = RYGate(x6).control(num_ctrl_qubits=1,ctrl_state=0).control(num_ctrl_qubits=1,ctrl_state=1)
rry7 = RYGate(x7).control(num_ctrl_qubits=1,ctrl_state=1).control(num_ctrl_qubits=1,ctrl_state=1)

circ = QuantumCircuit(3)
circ.ry(x1,0)
circ.cry(x2,control_qubit=0,target_qubit=1,ctrl_state=0)
circ.cry(x3,control_qubit=0,target_qubit=1,ctrl_state=1)
circ.append(rry4,[0,1,2])
circ.append(rry5,[0,1,2])
circ.append(rry6,[0,1,2])
circ.append(rry7,[0,1,2])

circ.save_statevector()
print(circ.draw())
backend = Aer.get_backend('aer_simulator')
t_circ = transpile(circ, backend)
qobj = assemble(t_circ)
job = backend.run(qobj)

result = job.result()
o = result.get_statevector(circ, decimals=10)

a=np.real(o)
print(a,'制备结果')
print(pro,'实际')





