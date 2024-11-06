# https://github.com/Namr/YAQCS/blob/65a286f05fc917a39462d23ab1ad368f84466e7c/BQSKit%20Extensions/examples/cx.py
#https://www.researchgate.net/figure/Two-qubit-modular-gates-a-Decomposition-of-the-CNOT-gate-The-geometric-phase-chij-of_fig2_299483242
import numpy as np
from qiskit import *
from qiskit import Aer

backend = Aer.get_backend('unitary_simulator')
qc = QuantumCircuit(2)
qc.cx(1,0) #MSB is control bit

#CNOT is its own inverse, see if MS-based CNOT works
qc.ry(np.pi/2,1)
qc.rxx(np.pi/2,1,0) #bug?? works but should be qc.rxx(np.pi/4,1,0)
qc.ry(-np.pi/2,1)
qc.rx(-np.pi/2,0)
#qc.rz(-np.pi/2,1) #bug: does not work -> global phase is divided by 2 in qiskit
                   #also should be qc.rz(-np.pi/2,1), see sdg.py
qc.p(-np.pi/2,1) #works
#qc.sdg(1) #works

#yes, we get ID matrix back

job = execute(qc, backend)
result = job.result()
print(result.get_unitary(qc, decimals=3))
