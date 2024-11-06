# https://github.com/anuragksv/QuantumLibrary/blob/9e1afd758384335109f231480047632bdf309efe/qulib/.ipynb_checkpoints/QuantumTeleportation-checkpoint.py
#For Circuit Creation, Measurement, and Simulation 
import qiskit
from qiskit.providers.aer import AerSimulator
from qiskit import Aer, assemble
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

#Bloch Sphere Visualization
import kaleidoscope.qiskit
from kaleidoscope import bloch_sphere
from qiskit.visualization import plot_bloch_multivector, plot_histogram

#To intailize the state of Qubit for Quantum Teleportation
from qiskit.extensions import Initialize
from qiskit.quantum_info import random_statevector

qq = QuantumRegister(1, name='qq')
qr = QuantumRegister(1, name='qr')
qs = QuantumRegister(1, name='qs')
cq = ClassicalRegister(1, name='cq')
cr = ClassicalRegister(1, name='cr')
qc = QuantumCircuit(qq, qr, qs, cq, cr)

states = []

def initialize_q():
    qq_state = random_statevector(2)
    qq_init= Initialize(qq_state)
    qq_init.label = "init_state"
    qc.append(qq_init, qq)
    qc.barrier()
    states.append(qq_state)
    print(qc.draw(output='mpl'))
    return qq_state


def bell_pair_generation():
    qc.h(qr)
    qc.cx(qr,qs)
    qc.barrier()
    return qc.draw(output='mpl')

def encode():
    qc.cx(qq,qr)
    qc.h(qq)
    qc.barrier()
    return qc.draw(output='mpl')

def measure():
    qc.measure(qq,cq)
    qc.measure(qr,cr)
    qc.barrier()
    return qc.draw(output='mpl')

def decode():
    qc.x(qs).c_if(cr, 1)
    qc.z(qs).c_if(cq, 1)
    return qc.draw(output='mpl')
        
def final_state():

    sim = Aer.get_backend('aer_simulator')
    qc.save_statevector()
    out_vector = sim.run(qc).result().get_statevector()
    states.append(out_vector)
    #return bloch_sphere(out_vector)
    #return plot_bloch_multivector(out_vector)
    return out_vector
    

initialize_q()
bell_pair_generation()
encode()
measure()
decode()
final_state()


for i in states:
    print(i,'\n#############################################\n')
    print(bloch_sphere(i).show())
qc.draw(output='mpl')