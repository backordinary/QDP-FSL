# https://github.com/benkoehlL/Qiskit_Playground/blob/ab232c9787bc12a6feddf3ff5f2bac3f9a801ab8/Grovers_algorithm.py
'''
This program is an implementation of Grover's algorithm for a two-qubit system
'''

import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, draw, show
import numpy as np
from qiskit import IBMQ, BasicAer, QuantumCircuit, QuantumRegister,\
                    ClassicalRegister, execute
from qiskit.providers.ibmq import least_busy
from qiskit.tools.visualization import plot_histogram
from qiskit.tools.monitor import job_monitor

def phase_oracle(circuit, register):
    circuit.cz(register[0], register[1]) # now looks for state |11>
    #circuit.x(register[0])  # with this looks for state |x0>    
    #circuit.x(register[1])  # with this looks for state |0x>  
    circuit.barrier()

def inversion_about_average(circuit, register):
    circuit.h(register)
    circuit.x(register)
    circuit.h(register[1])
    circuit.cx(register[0], register[1])
    circuit.h(register[1])
    circuit.x(register)
    circuit.h(register)

qr = QuantumRegister(2)
oracleCircuit = QuantumCircuit(qr)
phase_oracle(oracleCircuit, qr)
#oracleCircuit.draw(output='mpl')
#draw()
#show(block=True)

qAverage = QuantumCircuit(qr)
inversion_about_average(qAverage, qr)
#qAverage.draw(output='mpl')
#draw()
#show(block=True)

qr = QuantumRegister(2)
cr = ClassicalRegister(2)

groverCircuit = QuantumCircuit(qr, cr)
groverCircuit.h(qr)

phase_oracle(groverCircuit, qr)
inversion_about_average(groverCircuit, qr)

groverCircuit.measure(qr, cr)
#groverCircuit.draw(output='mpl')
#draw()
#show(block=True)

backend = BasicAer.get_backend('qasm_simulator')
shots = 1024
results = execute(groverCircuit, backend=backend, shots=shots).result()
answer = results.get_counts()
plot_histogram(answer)
draw()
show(block=True)

# run on a real quantum computer
'''
IBMQ.load_account()
provider = IBMQ.get_provider(group='open')
backend_lb = least_busy(provider.backends(simulator=False, operational=True))
print("Least busy backend: ", backend_lb)
backend = backend_lb
job_exp = execute(groverCircuit, backend=backend, shots=shots)

job_monitor(job_exp, interval = 2)
results = job_exp.result()
answer = results.get_counts(groverCircuit)
plot_histogram(answer)
draw()
show(block=True)
'''