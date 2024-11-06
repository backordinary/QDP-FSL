# https://github.com/francienbarkhof/Quantum_Group1_Project/blob/aca6f81578031b2b8854de1b379ab2e0bba6c917/quantum_teleporation.py
# Do the necessary imports
import numpy as np
from qiskit import *
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import IBMQ, Aer
#from qiskit.visualization import plot_histogram, plot_bloch_multivector, array_to_latex
from qiskit.extensions import Initialize
#from qiskit import marginal_counts
from qiskit.quantum_info import random_statevector
from qiskit.visualization import plot_bloch_multivector
from quantuminspire.qiskit import QI
from getpass import getpass

def get_authentication():
    """Gets the authentication for connecting to the
       Quantum Inspire API.
    """
    print('Enter email:')
    email = input()
    print('Enter password')
    password = getpass()
    return email, password 

def create_bell_pair(qc, a, b):
    """Creates a bell pair in qc using qubits a & b"""
    qc.h(a) # Put qubit a into state |+>
    qc.cx(a,b) # CNOT with a as control and b as target

def alice_gates(qc, psi, a):
    qc.cx(psi, a)
    qc.h(psi)

def measure_and_send(qc, a, b):
    """Measures qubits a & b and 'sends' the results to Bob"""
    qc.barrier()
    qc.measure(a,0)
    qc.measure(b,1)

# This function takes a QuantumCircuit (qc), integer (qubit)
# and ClassicalRegisters (crz & crx) to decide which gates to apply
def bob_gates(qc, qubit, crz, crx):
    # Here we use c_if to control our gates with a classical
    # bit instead of a qubit
    qc.x(qubit).c_if(crx, 1) # Apply gates if the registers 
    qc.z(qubit).c_if(crz, 1) # are in the state '1'

if __name__ == '__main__':
    if 'authentication' not in vars().keys():
        authentication = get_authentication()
    QI.set_authentication_details(*authentication)
    qi_backend = QI.get_backend('QX single-node simulator')

    psi = random_statevector(2)

    print(psi)

    init_gate = Initialize(psi)
    init_gate.label = "init"

    ## SETUP
    qr = QuantumRegister(3, name="q")   # Protocol uses 3 qubits
    crz = ClassicalRegister(1, name="crz") # and 2 classical registers
    crx = ClassicalRegister(1, name="crx")
    qc = QuantumCircuit(qr, crz, crx)

    ## STEP 0
    # First, let's initialize Alice's q0
    qc.append(init_gate, [0])
    qc.barrier()

    ## STEP 1
    # Now begins the teleportation protocol
    create_bell_pair(qc, 1, 2)
    qc.barrier()

    ## STEP 2
    # Send q1 to Alice and q2 to Bob
    alice_gates(qc, 0, 1)

    ## STEP 3
    # Alice then sends her classical bits to Bob
    measure_and_send(qc, 0, 1)

    ## STEP 4
    # Bob decodes qubits
    bob_gates(qc, 2, crz, crx)

    # Display the circuit
    print(qc)

    sim = Aer.get_backend('aer_simulator')
    qc.save_statevector()
    out_vector = sim.run(qc).result().get_statevector()
    print(out_vector)
    plot_bloch_multivector(out_vector)