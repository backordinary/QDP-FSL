# https://github.com/reverseredoc/SoC-2022/blob/16a5bc0412142b92407eea0ab5169948003a2d09/week2/bell_states.py
#bell states


# initialization
import numpy as np

# importing Qiskit
from qiskit import IBMQ, Aer
from qiskit.providers.ibmq import least_busy
from qiskit import QuantumCircuit, assemble, transpile

# import basic plot tools
from qiskit.visualization import plot_histogram


def bellState(n):
    one_state=[0,1]
    circuit=QuantumCircuit(2,2)
    if n==2:
        circuit.initialize(one_state,0)
    if n==3:
        circuit.initalize(one_state,1)
    if n==4:
        circuit.initialize(one_state,0)
        circuit.initialize(one_state,1)
         
    circuit.h(0)
    circuit.cx(0,1)   
    return circuit
  
    
circuit=bellState(1)
display(circuit.draw())
 

svsim = Aer.get_backend('aer_simulator')

circuit.save_statevector()
qobj = assemble(circuit)
final_state = svsim.run(qobj).result().get_statevector()

from qiskit.visualization import array_to_latex
array_to_latex(final_state, prefix="\\text{Statevector} = ")


        
    
    
