# https://github.com/Yanaishaviv/AvodatQS/blob/1158d4dd787be6024920a8243cea202b66dd51b1/cyber/quan.py
import numpy as np
from qiskit import IBMQ, BasicAer, circuit, execute, Aer
from qiskit.providers.ibmq import least_busy
from qiskit import QuantumCircuit, execute
from qiskit.pulse.schedule import draw
from qiskit.tools.jupyter import *
from ibm_quantum_widgets import draw_circuit
from qiskit.visualization import plot_histogram


circuit = QuantumCircuit(2, 2)

circuit.h(0)
circuit.cx(0, 1)
circuit.measure([0, 1], [0, 1])
print(circuit)
simualtor  = Aer.get_backend('qasm_simulator')
job = execute(circuit, simualtor)
result = job.result()

count = result.get_counts(circuit)
print(count)
plot_histogram(count)