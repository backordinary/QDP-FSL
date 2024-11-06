# https://github.com/3gaspo/guide-infoQ/blob/ae8ec94a5bfb715168017518abb4beb51c969713/codes/simulation_qbits.py
##dans le shell
#pip install qiskit
#pip install qiskit[visualization]

##dans le script
from qiskit import QuantumCircuit, execute, Aer

#Simulation d'un Qbit
n = 3
qc = QuantumCircuit(n,n) #n Qbits et n Cbits
for j in range(n):
    qc.measure(j,j) #ajout  d'une mesure du jième Qbit vers le jième Cbit
    
qc.draw() #Affiche le circuit construit

#Sens conventionnel
qc = QuantumCircuit(n,n) #n Qbits et n Cbits
for j in range(n-1,-1,-1):
    qc.measure(j,j) #ajout  d'une mesure du jième Qbit vers le jième Cbit
qc.draw() #Affiche le circuit construit


#Initialisation d'un circuit
from math import sqrt

qc = QuantumCircuit(n,n)

initial_state_0 = [0,1] #état |1>
initial_state_1 = [1,0] #état |0>
initial_state_2 = [(1/sqrt(2))*(1+1j),0] #état de même probabilité que |0>

initial_state = []
initial_state.append(initial_state_0)
initial_state.append(initial_state_1)
initial_state.append(initial_state_2)

for j in range(n):
    qc.initialize(initial_state[j], j) #initier le Qbit j à l'état initial_state_j

qc.measure(range(n-1,-1,-1),range(n-1,-1,-1)) #syntaxe + rapide et sens conventionnel
qc.draw()


#Mesure d'un Qbit avec qasm_simulator
backend = Aer.get_backend('qasm_simulator')
counts = execute(qc, backend).result().get_counts()

from qiskit.visualization import plot_histogram
plot_histogram(counts)


#Superposition d'états
qc = QuantumCircuit(2,2)
qc.initialize([1,0],0)
qc.initialize([1/sqrt(2),1/sqrt(2)],1)
qc.measure(1,1)
qc.measure(0,0)

counts = execute(qc, backend).result().get_counts()
plot_histogram(counts)


#Mesure d'un Qbit avec statevector_simulator
qc = QuantumCircuit(1) #juste un Qbit, pas besoin de Cbit
qc.initialize([0,1], 0)
backend = Aer.get_backend('statevector_simulator')
result = execute(qc,backend).result()
state = result.get_statevector()

print(state)
#Cela va renvoyer [0.+0.j 1.+0.j]

from qiskit_textbook.tools import array_to_latex
array_to_latex(state, pretext="\\text{Statevector} = ")
#Cette ligne de code ne fonctionne que sur un Jupyter notebook
#Cela  renvoie le ket associé.

from qiskit import assemble
qobj = assemble(qc) #Ceci est un nouveau type
state = backend.run(qobj).result().get_statevector()
print(state)


#Sphère de Bloch
#Nous allons voir deux syntaxes possibles
from qiskit_textbook.widgets import plot_bloch_vector_spherical
from qiskit.visualization import plot_bloch_multivector
from math import pi

qc = QuantumCircuit(2,2)
qc.initialize([1/sqrt(2),1/sqrt(2)],1)
state = execute(qc,Aer.get_backend('statevector_simulator')).result().get_statevector()

#Ne peut tracer qu'un seul Qbit
coords = [pi/2,0,1] # [Theta, Phi, Rayon]
plot_bloch_vector_spherical(coords)

#Plusieurs Qbits
plot_bloch_multivector(state)


#Qsphère
from qiskit.visualization import plot_state_qsphere
import numpy as np

#réécriture de 1/sqrt(2) (00+01), même type que get_statevector
state = state = np.array([1,0,1/sqrt(2),1/sqrt(2)])

plot_state_qsphere(state)