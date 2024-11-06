# https://github.com/woahdas/Programs-in-Quantum-Computing/blob/cca0e80371dc6233a3c17198b5de296fe8dbf988/FirstQiskitProgram.py
#This is my first ever program I ever ran relating to quantum computing, on an actual IBM quantum computer, alongside one of their simulators, and my own simulator.

import qiskit as q

circuit = q.QuantumCircuit(2,2) #2 qubits + 2 classical bits

#currently 0,0 for qubits
circuit.x(0) #apply a NOT gate to qubit 0
#now 1,0 for qubits(?)
circuit.cx(0, 1) #a control NOT or cNOT, flips 2nd qubit value if first qubit is 1 (entangles them(?)
#now 1,1
circuit.measure([0,1], [0,1]) #measure the qubits, now they will collapse into their values
                #^ ^, qubit 0 will be the value we're going to return for classical bit 0, same for 1 and 1(?)
                #when we take this measurement, we should get a value of 1,1 for/as classical bits


circuit.draw() #get an ASCII representation of our circuit
#you can also do 'circuit.draw(output="mpl")' to get a matplotlib representation

#running it on an actual quantum computer(?)
from qiskit import IBMQ
#IBMQ.save_account(open("token.txt", "r").read()) #only need to do this once

IBMQ.load_account() #using our account
provider = IBMQ.get_provider("ibm-q") #the provider communicates to the backend

for backend in provider.backends():
    try:
        qubit_count = len(backend.properties().qubits)
    except:
        qubit_count = "simulated" #doesn't look at simulated qubits
    print(f"{backend.name()} has {backend.status().pending_jobs} queued and {qubit_count} qubits")
    
from qiskit.tools.monitor import job_monitor #this will tell us where we are in the queue
backend = provider.get_backend("ibmq_athens")
job = q.execute(circuit, backend=backend, shots=500) #shots is how many times to run it(?)
job_monitor(job)

from qiskit.visualization import plot_histogram
from matplotlib import style #

style.use("dark_background") #needed if on a dark mode (dark background)

result = job.result()
counts = result.get_counts(circuit)

plot_histrogram([counts])

#running on a simulator
backend = provider.get_backend("ibmq_qasm_simulator")
circuit = q.QuantumCircuit(2,2) #2 qubits + 2 classical bits


circuit.h(0) #instead of flipping the other qubit, we're putting into a SUPERPOSITION
circuit.cx(0, 1) 
circuit.measure([0,1], [0,1])

circuit.draw()
#RUNNING ON A *LOCAL SIMULATOR*, VERY IMPORTANT, PROBABLY WHAT I'M ALWAYS GOING TO DO IN ALL HONESTY
from qiskit import Aer #simulator framework for qiskit
sim_backend = Aer.get_backend("qasm simulator")

job = q.execute(circuit, backend=sim_backend, shots=500) #shots is how many times to run it(?)
job_monitor(job)

result = job.result()
counts = result.get_counts(circuit)

plot_histrogram([counts])
