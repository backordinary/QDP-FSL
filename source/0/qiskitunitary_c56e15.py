# https://github.com/supratimece/BARCQ_Django/blob/911bd5a39b51ebfe2dc22a91c88485093e835050/backend/qiskitUnitary.py
import numpy as np
from qiskit import *
from qiskit import Aer

#Changing the simulator 
backend = Aer.get_backend('unitary_simulator')

#The circuit without measurement
circ = QuantumCircuit(2)
circ.crz(1.57079632,0,1)

#job execution and getting the result as an object
job = execute(circ, backend)
result = job.result()

#get the unitary matrix from the result object
print(result.get_unitary(circ, decimals=3))
