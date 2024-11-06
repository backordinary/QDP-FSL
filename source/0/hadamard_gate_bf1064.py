# https://github.com/v41bh4v/Qiskit_Tutorial/blob/c5741e8d1884f6e5cd6a6c88829d9845a9e3a820/Learn%20Qiskit%20step%20by%20step/Hadamard_gate.py

from qiskit import *

qc =QuantumCiruit()
#Quantum registers
qr = QuantumRegister(2,'qreg')
#Giving it a name like 'qreg' is optional.
#Now we can add it to the circuit using the add_register method, and see that it has been added by checking the qregs variable of the circuit object.
qc.add_register( qr )

qc.qregs
#We can see our circuit by draw()
qc.draw(output='mpl')

qc.h( qr[0] )