# https://github.com/JLee-Sin/QisKit-Practice-as-Python-files/blob/bdc1301304686c43701cbf3c3c7a12b212314830/Week%2015.py
import numpy as np
# Importing standard Qiskit libraries
from qiskit import QuantumCircuit, execute, transpile, Aer, IBMQ
from qiskit.tools.jupyter import *
from qiskit.visualization import *

# Loading your IBM Quantum account(s)
provider = IBMQ.load_account()
print('Libraries imported successfully!')

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.draw()

backend = Aer.get_backend('statevector_simulator')
job = execute(qc, backend)
result = job.result()
state = result.get_statevector()
array_to_latex(state, prefix="\\text{Statevector} = ")
counts = result.get_counts()
plot_histogram(counts)

q_c = QuantumCircuit(2)
q_c.h(0)
q_c.cx(0, 1)
q_c.x(1)
q_c.draw()

backend = Aer.get_backend('statevector_simulator')
job = execute(q_c, backend)
result = job.result()
state = result.get_statevector()
array_to_latex(state, prefix="\\text{Statevector} = ")

Qc = QuantumCircuit(2)
Qc.h(0)
Qc.cx(0, 1)
Qc.z(0)
Qc.x(0)
Qc.draw()

backend = Aer.get_backend('statevector_simulator')
job = execute(Qc, backend)
result = job.result()
state = result.get_statevector()
array_to_latex(state, prefix="\\text{Statevector} = ")

qc = QuantumCircuit(2)
qc.h(0)
qc.x(0)

state_choice = input("What bell state do you want")

if state_choice == "1":
    pass
elif state_choice == "2":
    qc.x(0)
elif state_choice == "3":
    qc.z(0)
elif state_choice == "4":
    qc.x(0)
    qc.z(0)
qc.draw()

q_c = QuantumCircuit(2)
q_c.h(0)
q_c.x(1)
q_c.x(0)
q_c.cx(0, 1)
q_c.draw()

backend = Aer.get_backend('statevector_simulator')
job = execute(q_c, backend)
result = job.result()
state = result.get_statevector()
array_to_latex(state, prefix="\\text{Statevector} = ")

qc = QuantumCircuit(2)
qc.h(1)
qc.cz(1, 0)
qc.draw()

backend = Aer.get_backend('statevector_simulator')
job = execute(qc, backend)
result = job.result()
state = result.get_statevector()
array_to_latex(state, prefix="\\text{Statevector} = ")
