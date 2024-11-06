# https://github.com/AlexsashaV4/bachelor_thesis_QEC/blob/ac70dbc21a25aac883de7deba4739bfe31bdf5be/QEC_Simulations/Rep_code.py
from qiskit import *
from qiskit import Aer, transpile
from qiskit.quantum_info.operators.symplectic.pauli import Pauli
from qiskit.visualization import plot_histogram, plot_state_paulivec, plot_state_hinton
from qiskit.visualization import plot_state_qsphere, plot_bloch_vector, plot_state_city, plot_bloch_multivector
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.tools.visualization import circuit_drawer
from qiskit.tools.monitor import job_monitor
from qiskit.providers.basicaer import QasmSimulatorPy
from qiskit import assemble
from qiskit.quantum_info import Statevector
from qiskit import QuantumCircuit, execute, Aer
from qiskit.visualization import plot_histogram
import qiskit.providers.aer.noise as noise
from qiskit.providers.aer.noise.errors import pauli_error, depolarizing_error
import numpy as np
import matplotlib.pyplot as plt 
from qiskit.providers.aer.noise import NoiseModel

def get_noise(p):

    error_meas = pauli_error([('X',p), ('I', 1 - p)])

    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_meas, "measure") # measurement error is applied to measurements
        
    return noise_model

noise_model = get_noise(0.01)
### Create the simulator
sim = Aer.get_backend('aer_simulator')

### Create the quantum circuit
qc_3qx = QuantumCircuit(3)

### Create the initial state \ket{\psi} (the information qubit)
initial_state = [1/np.sqrt(3), np.sqrt(2)/np.sqrt(3)]  
qc_3qx.initialize(initial_state, 0) # Initialize the 0th qubit in the state `initial_state`
qc_3qx.barrier()

###Encoding procedure
qc_3qx.cx(0,1)
qc_3qx.cx(0,2)
qc_3qx.barrier()

### Draw
#circuit_drawer(qc_3qx, output='mpl', fold=-1,style={'backgroundcolor': '#EEEEEE'},filename='my_circuit.png')
#plt.show()




###Plot on the q sphere
#qc_3qx.save_statevector()
#result = sim.run(transpile(qc_3qx, sim), shots=1000).result()
#psi  = result.get_statevector(qc_3qx)
#plot_state_qsphere(psi)
#plot_state_city(psi)
#plt.show()


##########
 #Define the error channel (still to do is implementing the probability)
##########


######
#Create the bitflip error
######
def apply_err(n, err):
    qc = QuantumCircuit(int(n), name='Error')####! ad probability
    which_qubit = np.random.randint(n)
    
    if err=='bit':
        qc.x(which_qubit)
    elif err=='phase':
        qc.z(which_qubit)
    else:
        pass
    err = qc.to_gate()
    
    return err, which_qubit

err, which_qubit = apply_err(3, 'bit')

#more general type of error

aer_sim = Aer.get_backend('aer_simulator')

####Create the 3bi-flip error correction code 
qc_3qx.append(err,range(3))
k=2
anc=QuantumRegister(k, 'auxiliary')
qc_3qx.add_register(anc)
cr=ClassicalRegister(k, 'syndrome')
qc_3qx.add_register(cr)
qc_3qx.cx(0,3)
qc_3qx.cx(1,3)
qc_3qx.cx(1,4)
qc_3qx.cx(2,4)
qc_3qx.barrier()
qc_3qx.measure(anc[0],cr[0])
qc_3qx.measure(anc[1],cr[1])

# ####Before error correction
# cr2=ClassicalRegister(3, 'outcome')
# qc_3qx.add_register(cr2)
# qc_3qx.measure([0,1,2],[2,3,4])
# qc_3qx.draw('mpl')
# counts=execute(qc_3qx,backend = sim, shots=1024).result().get_counts()
# plot_histogram(counts)
# plt.show()


###After error correction
qc_3qx.x(0).c_if(cr, 1)###condition is in binary
qc_3qx.x(1).c_if(cr, 3)
qc_3qx.x(2).c_if(cr, 2)
qc_3qx.barrier()
cr2=ClassicalRegister(3, 'outcome')
qc_3qx.add_register(cr2)
qc_3qx.measure([0,1,2],[2,3,4])
qc_3qx.draw('mpl')
counts=execute(qc_3qx,backend = aer_sim, noise_model= noise_model, shots=1024).result().get_counts()
plot_histogram(counts)
plt.show()