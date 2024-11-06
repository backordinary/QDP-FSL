# https://github.com/Mouhamedaminegarrach/Quantum-Send-File/blob/0754a07b3cc29ca596621f294944f23de199761d/QOSF%20project/NEQRtest/QTeleport.py
from qiskit import*
from qiskit.extensions import Initialize

def entanglement_bell_pair(qc, a, b):
    
    qc.h(a) # Put qubit a into state |+> or |-> using hadamard gate
    qc.cx(a,b) # CNOT with a as control and b as target
def alice_state_qubits(qc, psi, a):
    qc.cx(psi, a) #psi is the state of q0
    qc.h(psi)
def measure_classical_send(qc, a, b):
    
    qc.barrier()
    qc.measure(a,[0,1,2,3,4,5,6,7,8,9])
    qc.measure(b,[10,11,12,13,14,15,16, 17,18,19])
def bob_apply_gates(qc, qubit, cr1, cr2):

    qc.z(qubit).c_if(cr1, 1)  #if cr1 is 1 apply Z gate
    qc.x(qubit).c_if(cr2, 1) #if cr2 is 1 apply x gate, look at table above
    

def QTeleport(qc):
    backend = Aer.get_backend('statevector_simulator')
    job = execute(qc, backend)
    #print(f"print(job):{job}\n\n")
    #print(job.status())
    result = job.result()
    #print(f"print(result):{result}\n\n")
    output = result.get_statevector(qc)

    init_gate = Initialize(output)

    qr = QuantumRegister(30)   
    cr1 = ClassicalRegister(10) 
    cr2 = ClassicalRegister(10)
    qc = QuantumCircuit(qr, cr1, cr2)

    #let's initialise Alice's q0
    qc.append(init_gate, [0,1,2,3,4,5,6,7,8,9])
    qc.barrier()

    # teleportation protocol
    entanglement_bell_pair(qc, [10,11,12,13,14,15,16, 17,18,19], [20,21,22,23,24,25,26,27,28,29])
    qc.barrier()
    # Send q1 to Alice and q2 to Bob
    alice_state_qubits(qc, [0,1,2,3,4,5,6,7,8,9], [10,11,12,13,14,15,16, 17,18,19])

    # alice sends to Bob
    measure_classical_send(qc, [0,1,2,3,4,5,6,7,8,9], [10,11,12,13,14,15,16, 17,18,19])

    # Bob decodes qubits
    bob_apply_gates(qc, [20,21,22,23,24,25,26,27,28,29], cr1, cr2)

    inverse_init_gate = init_gate.gates_to_uncompute()
    qc.append(inverse_init_gate, [20,21,22,23,24,25,26,27,28,29])

    cr_result = ClassicalRegister(10)
    qc.add_register(cr_result)
    qc.measure([20,21,22,23,24,25,26,27,28,29],[20,21,22,23,24,25,26,27,28,29])
    return(qc)