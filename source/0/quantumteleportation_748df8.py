# https://github.com/anuragksv/QuantumLibrary/blob/9e1afd758384335109f231480047632bdf309efe/qulib/QuantumTeleportation.py
#For Circuit Creation, Measurement, and Simulation 
from qiskit import Aer, assemble
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

#Bloch Sphere Visualization
from qiskit.visualization import plot_bloch_multivector, plot_histogram

#To intailize the state of Qubit for Quantum Teleportation
from qiskit.extensions import Initialize
from qiskit.quantum_info import random_statevector

def QuantumTeleportation():
    qq = QuantumRegister(1, name='qq')
    qr = QuantumRegister(1, name='qr')
    qs = QuantumRegister(1, name='qs')
    cq = ClassicalRegister(1, name='cq')
    cr = ClassicalRegister(1, name='cr')
    qc = QuantumCircuit(qq, qr, qs, cq, cr)
    
    psi = random_statevector(2)
    qq_init= Initialize(psi)
    qq_init.label = "init_state"
    
    def initialize_psi():
        qc.append(qq_init, qq)
        qc.barrier()
        return plot_bloch_multivector(psi)
    
    
    def bell_pair_generation():
        qc.h(qr)
        qc.cx(qr,qs)
        qc.barrier()
        #return qc.draw(output='mpl')
    
    def encode():
        qc.cx(qq,qr)
        qc.h(qq)
        qc.barrier()
        #return qc.draw(output='mpl')
    
    def measure():
        qc.measure(qq,cq)
        qc.measure(qr,cr)
        qc.barrier()
        #return qc.draw(output='mpl')
    
    def decode():
        qc.x(qs).c_if(cr, 1)
        qc.z(qs).c_if(cq, 1)
        #return qc.draw(output='mpl')
        
    def final_states():
        sim = Aer.get_backend('aer_simulator')
        qc.save_statevector()
        out_vector = sim.run(qc).result().get_statevector()
        return plot_bloch_multivector(out_vector)
    
    def disentangle():
        inverse_qq_init = qq_init.gates_to_uncompute()
        qc.append(inverse_qq_init, qs)

    initialize_psi()
    bell_pair_generation()
    encode()
    measure()
    decode()
    final_states()
    disentangle()
    qc.draw(output='mpl')