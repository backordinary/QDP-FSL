# https://github.com/andrei-saceleanu/Quantum_Computing/blob/768ed606975a6bd62c07242f4ba5151ff8e1196f/teleportation.py
from qiskit import QuantumCircuit,execute,Aer,QuantumRegister, ClassicalRegister
from qiskit.visualization import plot_histogram
from qiskit.extensions import Initialize
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt,pi

def create_bell_pair(qc,a,b):
	qc.h(a)
	qc.cx(a,b)


def alice_gates(qc,psi,a):
	qc.cx(psi,a)
	qc.h(psi)

def measure_send(qc,a,b):
	qc.measure(a,0)
	qc.measure(b,1)

def bob_gates(qc,qubit,crz,crx):
	qc.x(qubit).c_if(crx,1)
	qc.z(qubit).c_if(crz,1)

#Initialize quantum state
psi=np.random.rand(1,2)
psi=psi/np.linalg.norm(psi)
psi=psi[0,:]
init_gate=Initialize(psi)

#Initialize circuit
qr=QuantumRegister(3,name='q')
crz=ClassicalRegister(1,name='crz')
crx=ClassicalRegister(1,name='crx')

circuit=QuantumCircuit(qr,crz,crx)

circuit.append(init_gate,[0])
circuit.barrier()

create_bell_pair(circuit, 1, 2)
circuit.barrier()

alice_gates(circuit, 0, 1)
circuit.barrier()

measure_send(circuit, 0, 1)
circuit.barrier() 

bob_gates(circuit, 2, crz, crx)
circuit.barrier()

inverse=init_gate.gates_to_uncompute()
circuit.append(inverse,[2])

#Final measure
cr_res=ClassicalRegister(1)
circuit.add_register(cr_res)
circuit.measure(2,2)
circuit.draw('mpl')

#Simulation part
backend = Aer.get_backend('qasm_simulator')
counts = execute(circuit, backend, shots=1024).result().get_counts()
plot_histogram(counts)
plt.show()


