# https://github.com/DanilShayakhmetov/graduate_project/blob/25ee4b26e68ed6c587a02d607cdf69ebc216da3e/Grover.py

from qiskit import QuantumRegister, ClassicalRegister
from qiskit import QuantumCircuit, Aer, execute

q1 = QuantumRegister(10)
c1 = ClassicalRegister(10)
qc1 = QuantumCircuit(q1, c1)

# qc1.h(q[0])
# qc1.cx(q[0], q[1])
# qc1.measure(q, c)




#Make alghoritm

# qc1.x(q1[0])
# qc1.x(q1[5])
#

qc1.x(q1[1])
qc1.x(q1[6])



qc1.h(q1[1])
qc1.h(q1[6])
qc1.h(q1[9])
qc1.x(q1[1])
qc1.ccx(q1[0], q1[1], q1[2])
qc1.x(q1[0])
qc1.x(q1[1])
qc1.ccx(q1[0], q1[1], q1[3])
qc1.x(q1[0])

qc1.x(q1[6])
qc1.ccx(q1[5], q1[6], q1[7])
qc1.x(q1[5])
qc1.x(q1[6])
qc1.ccx(q1[5], q1[6], q1[8])
qc1.x(q1[5])

qc1.ccx(q1[2], q1[3], q1[4])
qc1.ccx(q1[7], q1[4], q1[2])
qc1.ccx(q1[8], q1[4], q1[3])

#---------------


qc1.ccx(q1[3], q1[4], q1[9])

#---------------

qc1.h(q1[1])
qc1.h(q1[6])
qc1.h(q1[9])
qc1.x(q1[1])
qc1.ccx(q1[0], q1[1], q1[2])
qc1.x(q1[0])
qc1.x(q1[1])
qc1.ccx(q1[0], q1[1], q1[3])
qc1.x(q1[0])

qc1.x(q1[6])
qc1.ccx(q1[5], q1[6], q1[7])
qc1.x(q1[5])
qc1.x(q1[6])
qc1.ccx(q1[5], q1[6], q1[8])
qc1.x(q1[5])

qc1.ccx(q1[2], q1[3], q1[4])
qc1.ccx(q1[7], q1[4], q1[2])
qc1.ccx(q1[8], q1[4], q1[3])



qc1.h(q1[1])
qc1.x(q1[1])

qc1.h(q1[6])
qc1.x(q1[6])
qc1.h(q1[6])

qc1.cx(q1[1], q1[6])

qc1.x(q1[1])
qc1.h(q1[1])

qc1.h(q1[6])
qc1.x(q1[6])
qc1.h(q1[6])

qc1.h(q1[9])


# #Measure
# for i in range(10):
#     qc1.measure(q1[i], c1[i])


qc1.measure(q1[1], c1[1])
qc1.measure(q1[6], c1[6])

backend = Aer.get_backend('qasm_simulator')
job_sim = execute(qc1, backend)
sim_result = job_sim.result()

print(sim_result.get_counts(qc1))