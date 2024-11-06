# https://github.com/ebrutpc/Quantum-Computing/blob/881e06cca4b711bbd04d699c5235bd53c6570fa4/6Qubits-QFT.py


import numpy as np
from numpy import pi
from qiskit import QuantumCircuit , execute,Aer,IBMQ,QuantumRegister,ClassicalRegister
from qiskit.visualization import plot_histogram


q = QuantumRegister(6)
c  = ClassicalRegister(6)
circuit = QuantumCircuit(q,c)


circuit.h(q[0])
circuit.cp(np.pi/2,q[1],q[0])
circuit.cp(np.pi/4,q[2],q[0])
circuit.cp(np.pi/8,q[3],q[0])
circuit.cp(np.pi/16,q[4],q[0])
circuit.cp(np.pi/32,q[5],q[0])
circuit.barrier()
circuit.h(q[1])
circuit.cp(np.pi/2,q[2],q[1])
circuit.cp(np.pi/4,q[3],q[1])
circuit.cp(np.pi/8,q[4],q[1])
circuit.cp(np.pi/16,q[5],q[1])
circuit.barrier()
circuit.h(q[2])
circuit.cp(np.pi/2,q[3],q[2])
circuit.cp(np.pi/4,q[4],q[2])
circuit.cp(np.pi/8,q[5],q[2])
circuit.barrier()
circuit.h(q[3])
circuit.cp(np.pi/2,q[4],q[3])
circuit.cp(np.pi/4,q[5],q[3])
circuit.barrier()
circuit.h(q[4])
circuit.cp(np.pi/2,q[5],q[4])
circuit.barrier()
circuit.h(q[5])


circuit.draw(output='mpl',fold=0)

circuit.measure(q,c)


simulator =Aer.get_backend('qasm_simulator')
job = execute(circuit, backend=simulator,shots=8192)
result = job.result()
counts = result.get_counts(circuit)
plot_histogram(counts)

