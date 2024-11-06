# https://github.com/AlexsashaV4/bachelor_thesis_QEC/blob/d7b731fec1abc3dec6b73ea3a1677031db44fcad/QEC_Simulations/Seven_qubit_code.py
from unicodedata import name
from qiskit import *
from qiskit import Aer, transpile
from qiskit.providers.aer import AerSimulator
from qiskit.quantum_info.operators.symplectic.pauli import Pauli
from qiskit.visualization import plot_histogram, plot_state_paulivec, plot_state_hinton
from qiskit.visualization import plot_state_qsphere, plot_bloch_vector, plot_state_city, plot_bloch_multivector
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.tools.visualization import circuit_drawer
from qiskit.tools.monitor import job_monitor
from qiskit.providers.basicaer import QasmSimulatorPy
from qiskit import assemble
from qiskit.quantum_info import Statevector, partial_trace
from qiskit import QuantumCircuit, execute, Aer
from qiskit.visualization import plot_histogram
import qiskit.providers.aer.noise as noise
from qiskit.providers.aer.noise.errors import pauli_error, depolarizing_error
import numpy as np
import matplotlib.pyplot as plt 
from qiskit.providers.aer.noise import NoiseModel
import qiskit.quantum_info as qi


aer_sim = Aer.get_backend('aer_simulator')
sv_sim = Aer.get_backend('statevector_simulator')
#result = execute(qc_3qx, sv_sim).result()
#sv = result.get_statevector()

######
# Create a custum gate that is affected by error 
######
all_qubits=[0,1,2,3,4,5,6]
qe=QuantumCircuit(7, name='error')
qe2=QuantumCircuit(7, name='error2')
ident3 = qi.Operator(np.identity(2**7))
qe.unitary(ident3, all_qubits, label='error')
qe2.unitary(ident3, all_qubits, label='error2')
#######
# Create the quantum circuit with the initial state
#######
qc_3qx = QuantumCircuit(7)
#initial_state = [1/np.sqrt(3), np.sqrt(2)/np.sqrt(3)] 
#initial_state = [0,1] 
#qc_3qx.initialize(initial_state, 3)
# # Initialize the 0th qubit in the state `initial_state`

qc_3qx.barrier()

###
#Encoding
###
def Encoding7():
    q_encoding=QuantumCircuit(7,name='Enc')
    #Hadamards
    q_encoding.h(6)
    q_encoding.h(5)
    q_encoding.h(4)
    q_encoding.cx(0,1)
    q_encoding.cx(0,2)
    q_encoding.cx(6,3)
    q_encoding.cx(6,1)
    q_encoding.cx(6,0)
    q_encoding.cx(5,3)
    q_encoding.cx(5,2)
    q_encoding.cx(5,0)
    q_encoding.cx(4,3)
    q_encoding.cx(4,2)
    q_encoding.cx(4,1)
    #q_encoding.draw('mpl',  filename='Encoding_Seven')
    #plt.show()
    return q_encoding

#####
# Measurement function (Apply it when you want to look at the results)
#####
def measurement7(circ,nqubit,noise_model=NoiseModel(['unitary']),shots=100000):
    cr3=ClassicalRegister(nqubit, 'outcomes')
    circ.add_register(cr3)
    circ.measure(all_qubits,cr3)
    counts=execute(qc_3qx,backend = aer_sim, noise_model=noise_model, shots=shots).result().get_counts()
    return counts

qc_3qx.append(Encoding7(),all_qubits)

####
#Attaching unitary identity gates (faulty gates)
####
qc_3qx.append(qe, all_qubits)
qc_3qx.append(qe2, all_qubits)
qc_3qx.barrier()
# #####
# #Create the error model
# #####
def get_noise(p_error):
    #Bit flip error
    bit_flip = pauli_error([('X', p_error), ('I', 1 - p_error)])
    bit_flip1=bit_flip.tensor(bit_flip)
    bit_flip2=bit_flip.tensor(bit_flip1)
    bit_flip3=bit_flip.tensor(bit_flip2)
    bit_flip4=bit_flip.tensor(bit_flip3)
    bit_flip5=bit_flip.tensor(bit_flip4)
    bit_flip6=bit_flip.tensor(bit_flip5)

    #Phase flip error
    phase_flip = pauli_error([('Z', p_error), ('I', 1 - p_error)])
    phase_flip1=phase_flip.tensor(phase_flip)
    phase_flip2=phase_flip.tensor(phase_flip1)
    phase_flip3=phase_flip.tensor(phase_flip2)
    phase_flip4=phase_flip.tensor(phase_flip3)
    phase_flip5=phase_flip.tensor(phase_flip4)
    phase_flip6=phase_flip.tensor(phase_flip5)
    

    # ######
    # #Set error basis
    # ######
    noise_model = NoiseModel(['unitary'])
    noise_model.add_all_qubit_quantum_error(bit_flip6, 'error')
    #noise_model.add_all_qubit_quantum_error(phase_flip6, 'error2')
    return noise_model
######
# Measurement just after the noise, to se the effect of the noise
# uncomment these lines 
# #######

#counts = measurement7(qc_3qx, 7, noise_model=get_noise(0.01))
#plot_histogram(counts)
#plt.show()


#######
#Ancilla register: 
#AncX to correct X errors
#AncZ to correct Z errors
#######
ancX=QuantumRegister(3, 'anc_X') ### Ancilla qubit
qc_3qx.add_register(ancX)
ancZ=QuantumRegister(3, 'anc_Z') ### Ancilla qubit
qc_3qx.add_register(ancZ)
qc_3qx.h(7)
qc_3qx.h(8)
qc_3qx.h(9)
crX=ClassicalRegister(3, 'synd_X') ### Classical register for syndrome extraction
crZ=ClassicalRegister(3, 'synd_Z') ### Classical register for syndrome extraction
qc_3qx.add_register(crX)
qc_3qx.add_register(crZ)


#####
#Create the stabilizer
#####
stab1Z=QuantumCircuit(5, name='M1') #IIIZZZZ
for i in range(0,stab1Z.num_qubits-1):
    stab1Z.cx(i,4)
stab2Z=QuantumCircuit(5, name='M2') #IZZIIZZ
for i in range(0,stab2Z.num_qubits-1):
    stab2Z.cx(i,4)
stab3Z=QuantumCircuit(5, name='M3') #ZIZIZIZ
for i in range(0,stab3Z.num_qubits-1):
    stab3Z.cx(i,4)
stab1X=QuantumCircuit(5,name='M4') #IIIXXXX
for i in range(0,stab1X.num_qubits-1):
    stab1X.cx(4,i)
stab2X=QuantumCircuit(5, name='M5') #IXXIIXX
for i in range(0,stab2X.num_qubits-1):
    stab2X.cx(4,i)
stab3X=QuantumCircuit(5, name='M6') #XIXIXIX
for i in range(0,stab3X.num_qubits-1):
    stab3X.cx(4,i)


qc_3qx.append(stab1Z,[3,4,5,6,9])
qc_3qx.append(stab2Z,[1,2,5,6,8])
qc_3qx.append(stab3Z,[0,2,4,6,7]) ##Stab Z goes in ancX and then crX

qc_3qx.append(stab3X,[0,2,4,6,10])
qc_3qx.append(stab2X,[1,2,5,6,11])
qc_3qx.append(stab1X,[3,4,5,6,12]) ##Stab X goes in ancZ and then crZ
qc_3qx.h(7)
qc_3qx.h(8)
qc_3qx.h(9)
qc_3qx.barrier()
#Measure the ancilla results 
qc_3qx.measure(ancX[1],crX[1])
qc_3qx.measure(ancX[1],crX[1])
qc_3qx.measure(ancX[2],crX[2])

qc_3qx.measure(ancZ[0],crZ[0])
qc_3qx.measure(ancZ[1],crZ[1])
qc_3qx.measure(ancZ[2],crZ[2])
qc_3qx.barrier()

#### Uncomment these lines to see the ancilla values 
#qc_3qx.draw('mpl',scale=0.5)
#plt.show()
#counts=execute(qc_3qx, aer_sim, shots=10000).result().get_counts()
#plot_histogram(counts)
#plt.show()
# counts=execute(qc_3qx, aer_sim, noise_model= noise_model, shots=10000).result().get_counts()
# plot_histogram(counts)
# plt.show()

# ###
# #Recovery 
# ###
qc_3qx.barrier()
qc_3qx.z(0).c_if(crZ,0+1)
qc_3qx.x(0).c_if(crX,0+1) 
qc_3qx.z(1).c_if(crZ,1+1)
qc_3qx.x(1).c_if(crX,1+1) 
qc_3qx.z(2).c_if(crZ,2+1)
qc_3qx.x(2).c_if(crX,2+1) 
qc_3qx.z(3).c_if(crZ,3+1)
qc_3qx.x(3).c_if(crX,3+1) 
qc_3qx.z(4).c_if(crZ,4+1)
qc_3qx.x(4).c_if(crX,4+1) 
qc_3qx.z(5).c_if(crZ,5+1)
qc_3qx.x(5).c_if(crX,5+1) 
qc_3qx.z(6).c_if(crZ,6+1)
qc_3qx.x(6).c_if(crX,6+1) 
#qc_3qx.draw('mpl')
#plt.show()
#counts=execute(qc_3qx, aer_sim, noise_model= get_noise(0.1), shots=10000).result().get_counts()
#plot_histogram(counts)
#plt.show()



# #####
# # Decoding
# #####
qc_3qx.append(Encoding7().inverse(),all_qubits)
cr3=ClassicalRegister(7, 'outcomes')
qc_3qx.add_register(cr3)
qc_3qx.measure(all_qubits,cr3)
#qc_3qx.draw('mpl')
# # counts=execute(qc_3qx, aer_sim, shots=10000).result().get_counts()
# # plot_histogram(counts)
#plt.show()

# ###
# # Simulation
# ###
# #qc_3qx.draw('mpl',scale=0.5)
counts=execute(qc_3qx,backend = aer_sim, noise_model= get_noise(0.1), shots=10000).result().get_counts()
fig = plot_histogram(counts)
ax = fig.axes[0]
ax.set_xticklabels(ax.get_xticklabels(),fontsize='7')
#fig.savefig('output.png') # Or whatever you doing to output the image
#plot_histogram(counts,number_to_keep=10, sort='value_desc')
print(counts)
plt.show()