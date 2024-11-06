# https://github.com/daedalus/IBM_Quantum_Experiments/blob/680b02da129815c8df42f11e3ce281c6591eb62e/oracle.py
# coding: utf-8
#https://www.youtube.com/watch?v=LSA3pYZtRGg

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit

q = QuantumRegister('q',4)
tmp = QuantumRegister('tmp',1)
res = ClassicalRegister('res',4)

s = 14 # 1110

oracle = QuantumCircuit(q,tmp,res)
        
for i in range(len(q)):
    if (s & (1<<i)):
           oracle.cx(q[i],tmp[0])
           
bv = QuantumCircuit(q,tmp,res)
bv.x(tmp[0])
bv.h(q)
bv.h(tmp)
bv += oracle
bv.h(q)
bv.h(tmp)

bv.measure(q,res)
print(bv.qasm())

from qiskit.tools.visualization import circuit_drawer
circuit_drawer(bv)

from qiskit import QuantumProgram
qp = QuantumProgram()
qp.add_circuit(quantum_circuit=bv,name='bv')

from qiskit import backends
print(backends.local_backends())
print(backends.remote_backends())

import Qconfig
qp.set_api(Qconfig.APItoken,Qconfig.config['url'])

result = qp.execute('bv',backend='ibmqx4',timeout=3600)

counts = result.get_counts('bv')
print(counts)
from qiskit.tools.visualization import plot_histogram
plot_histogram(counts)
