# https://github.com/ahmedkfu2020/-/blob/a205805a9dfaef2f8cb2ff0645c597b1b119747c/ahmed20_2.py
# we can print the values by using python

print( round(-3**0.5/(2*2**0.5),4) )
print( round(3**0.5/(2*2**0.5),4) )
print( round(-1/(2*2**0.5),4) )
print( round(1/(2*2**0.5),4) )

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute, Aer
from math import pi

qreg =  QuantumRegister(2)
creg = ClassicalRegister(2)
mycircuit = QuantumCircuit(qreg,creg)

mycircuit.ry(2*(pi/6),qreg[1])
mycircuit.ry(2*(3*pi/4),qreg[0])

job = execute(mycircuit,Aer.get_backend('statevector_simulator'))
current_quantum_state=job.result().get_statevector(mycircuit)

for amplitude in current_quantum_state:
    print( round(amplitude.real,4) )
