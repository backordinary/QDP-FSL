# https://github.com/reverseredoc/SoC-2022/blob/16a5bc0412142b92407eea0ab5169948003a2d09/week2/teleportationtry.py
#teleportation
#this is the algorithm I tried using the other similar circuit for teleportation in which the measurements are shifted to the end of the circuit
# Do the necessary imports
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import IBMQ, Aer, transpile, assemble
from qiskit.visualization import plot_histogram, plot_bloch_multivector, array_to_latex
from qiskit.extensions import Initialize
from qiskit.ignis.verification import marginal_counts
from qiskit.quantum_info import random_statevector
from math import sqrt, pi
from qiskit.visualization import array_to_latex

circuit=QuantumCircuit(3,2)

#can be initilialised to any normalised state vector

state_vector=random_statevector(2)

display(array_to_latex(state_vector, prefix="\\text{State vector of psi} = "))#state vector of psi
circuit.initialize(state_vector,0)

display(plot_bloch_multivector(state_vector))

circuit.h(1)
circuit.cx(1,2)
circuit.barrier()
circuit.cx(0,1)
circuit.h(0)
circuit.barrier()
circuit.cx(1,2)
circuit.cz(0,2)

#using the quatum principle that any measurement can be shifted to the end of the circuit

circuit.measure(0,0)
circuit.measure(1,1)


display(circuit.draw())



circuit.save_statevector()
qobj = assemble(circuit)
final_state = svsim.run(qobj).result().get_statevector()


#checking the accuracy of the teleportation system
#qubit b in actual scenario now has the state psi
#as qubit 0 and 1 have been measured
a=[]
if final_state[0]!=0:
    a.append(final_state[0])
    break
if final_state[1]!=0:
    a.append(final_state[1])
    break
if final_state[2]!=0:
    a.append(final_state[2])
    break
if final_state[3]!=0:
    a.append(final_state[3])
    break
if final_state[4]!=0:
    a.append(final_state[4])
    break
if final_state[5]!=0:
    a.append(final_state[5])
    break
if final_state[6]!=0:
    a.append(final_state[6])
    break
if final_state[7]!=0:
    a.append(final_state[7])
    break

display(array_to_latex(a, prefix="\\text{State vector of b} = "))#state vector of b



#plotting bloch vectors
#verification 2
sim = Aer.get_backend('aer_simulator')
out_vector = sim.run(circuit).result().get_statevector()
plot_bloch_multivector(out_vector)
