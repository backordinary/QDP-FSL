# https://github.com/Abhigyan-Mishra/HHL/blob/330b06e7a6bdbdc3154c78c3e485834eff62a09e/hhl_qiskit.py
from qiskit import QuantumCircuit, execute,IBMQ
from math import pi
IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q')
qc = QuantumCircuit(7, 2)

#Initially applying Hadamard gate
qc.h(1)
qc.h(2)
qc.h(3)
qc.h(4)
qc.h(5)
qc.h(6)

#application of exp(iAt/16) operator
qc.h(6)
qc.ccx(1,5,6)
qc.h(6)
qc.cu3(0.196,-pi/2,pi/2,1,6) 
qc.cu3(pi/2,pi/2,-pi/2,1,6)
qc.u1(0.379,1)

qc.cu3(0.981,-pi/2,pi/2,1,5)
qc.u1(0.589,1)
qc.ccx(1,5,6)
qc.cu3(0.196,-pi/2,pi/2,1,5)
qc.ccx(1,5,6)
qc.h(6)
qc.ccx(1,5,6)
qc.h(6)

#application of exp(iAt/8) operator
qc.h(6)
qc.ccx(2,5,6)
qc.h(6)
qc.cu3(1.963,-pi/2,pi/2,2,6) 
qc.cu3(pi/2,pi/2,-pi/2,2,6)
qc.u1(1.115,2)

qc.cu3(1.963,-pi/2,pi/2,2,5)
qc.u1(2.615,2)
qc.ccx(2,5,6)
qc.cu3(0.178,-pi/2,pi/2,2,5)
qc.ccx(2,5,6)
qc.h(6)
qc.ccx(2,5,6)
qc.h(6)

#application of exp(iAt/4) operator
qc.h(6)
qc.ccx(3,5,6)
qc.h(6)
qc.cu3(-0.785,-pi/2,pi/2,3,6) 
qc.cu3(pi/2,pi/2,-pi/2,3,6)
qc.u1(1.017,3)

qc.cu3(3.927,-pi/2,pi/2,3,5)
qc.u1(2.517,3)
qc.ccx(3,5,6)
qc.cu3(2.356,-pi/2,pi/2,3,5)
qc.ccx(3,5,6)
qc.h(6)
qc.ccx(3,5,6)
qc.h(6)

#application of exp(iAt/2) operator
qc.h(6)
qc.ccx(4,5,6)
qc.h(6)
qc.cu3(-9.014*10**(-9),-pi/2,pi/2,4,6) 
qc.cu3(pi/2,pi/2,-pi/2,4,6)
qc.u1(-0.750,4)

qc.cu3(1.571,-pi/2,pi/2,4,5)
qc.u1(0.750,4)
qc.ccx(4,5,6)
qc.cu3(-1.571,-pi/2,pi/2,4,5)
qc.ccx(4,5,6)
qc.h(6)
qc.ccx(4,5,6)
qc.h(6)

#Applying Inverse Fourier Transform
qc.h(1)
qc.cu1(pi/2,2,1)
qc.cu1(pi/4,3,1)
qc.cu1(pi/8,4,1)
qc.h(2)
qc.cu1(pi/2,3,2)
qc.cu1(pi/4,4,2)
qc.h(3)
qc.cu1(pi/2,4,3)
qc.h(4)


#applying control-Ry operations
qc.cu3(8*pi/2,0,0,1,0)
qc.cu3(4*pi/2,0,0,2,0)
qc.cu3(2*pi/2,0,0,3,0)
qc.cu3(pi/2,0,0,4,0)

#Applying Quantum Fourier Transform
qc.h(4)
qc.cu1(-pi/2,4,3)
qc.h(3)
qc.cu1(-pi/4,4,2)
qc.cu1(-pi/2,3,2)
qc.h(2)
qc.cu1(-pi/8,4,1)
qc.cu1(-pi/4,3,1)
qc.cu1(-pi/2,2,1)
qc.h(1)

#application of exp(-iAt/2) operator
qc.h(6)
qc.ccx(4,5,6)
qc.h(6)
qc.ccx(4,5,6)
qc.cu3(-1.571,-pi/2,pi/2,4,5)
qc.ccx(4,5,6)
qc.u1(0.750,4)
qc.cu3(1.571,-pi/2,pi/2,4,5)
qc.u1(-0.750,4)
qc.cu3(pi/2,pi/2,-pi/2,4,6)
qc.cu3(-9.014*10**(-9),-pi/2,pi/2,4,6) 
qc.h(6)
qc.ccx(4,5,6)
qc.h(6)



#application of exp(-iAt/4) operator

qc.h(6)
qc.ccx(3,5,6)
qc.h(6)
qc.ccx(3,5,6)
qc.cu3(2.356,pi/2,-pi/2,3,5)
qc.ccx(3,5,6)
qc.u1(-2.517,3)
qc.cu3(3.927,pi/2,-pi/2,3,5)
qc.u1(-1.017,3)
qc.cu3(pi/2,3*pi/2,-3*pi/2,3,6)
qc.cu3(-0.785,pi/2,-pi/2,3,6)
qc.h(6)
qc.ccx(3,5,6)
qc.h(6)

#application of exp(-iAt/8) operator

qc.h(6)
qc.ccx(2,5,6)
qc.h(6)
qc.ccx(2,5,6)
qc.cu3(0.178,pi/2,-pi/2,2,5)
qc.ccx(2,5,6)
qc.u1(-2.615,2)
qc.cu3(1.963,pi/2,-pi/2,2,5)
qc.u1(-1.115,2)
qc.cu3(pi/2,3*pi/2,-3*pi/2,2,6)
qc.cu3(1.963,pi/2,-pi/2,2,6)
qc.h(6)
qc.ccx(2,5,6)
qc.h(6)



#Applying exp(-iAt/16) operator
qc.h(6)
qc.ccx(1,5,6)
qc.h(6)
qc.ccx(1,5,6)
qc.cu3(0.196,pi/2,-pi/2,1,5)
qc.ccx(1,5,6)
qc.u1(0.589,1)
qc.cu3(0.981,pi/2,-pi/2,1,5)
qc.u1(0.379,1)
qc.cu3(pi/2,3*pi/2,-3*pi/2,1,6)
qc.cu3(0.196,pi/2,-pi/2,1,6)
qc.h(6)
qc.ccx(1,5,6)
qc.h(6)


#At the end applying Hadamard gates
qc.h(1)
qc.h(2)
qc.h(3)
qc.h(4)

qc.measure([5,6], [0,1])

# get a backend
backend = provider.get_backend('ibmq_qasm_simulator')
# execute circuit
job=execute(qc, backend, shots=8192)
counts = job.result().get_counts()
print('RESULT: ',counts,'\n')