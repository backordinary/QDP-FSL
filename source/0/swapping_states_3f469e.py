# https://github.com/reverseredoc/SoC-2022/blob/16a5bc0412142b92407eea0ab5169948003a2d09/week2/swapping_states.py
#Swapping states

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import IBMQ, Aer, transpile, assemble
from qiskit.visualization import plot_histogram, plot_bloch_multivector, array_to_latex
from qiskit.extensions import Initialize
from qiskit.ignis.verification import marginal_counts
from qiskit.quantum_info import random_statevector
from math import sqrt, pi
from qiskit.visualization import array_to_latex


circuit=QuantumCircuit(2,2)

a=0#represents state vector of first qubit
b=1#represents state vector of second qubit
if a==1:
    circuit.x(0)
  
    
if b==1:
    circuit.x(1)
if a+b!=0:
    circuit.barrier()
 
circuit.cx(0,1)
circuit.cx(1,0)
circuit.cx(0,1)

display(circuit.draw())
   

sim = Aer.get_backend('aer_simulator')
circuit.save_statevector()
qobj = assemble(circuit)
final_state = sim.run(qobj).result().get_statevector()

display(array_to_latex(final_state, prefix="|\\psi\\rangle ="))
    
