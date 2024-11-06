# https://github.com/Talkal13/Quantum/blob/ccda55776da0a3f5bd212a8566f0a1e367061a6f/QCP/algorithms/grover.py
from qiskit import QuantumCircuit, QuantumRegister
from math import sqrt, pi, ceil

def Grover(n, O, it=0):
    '''Executes groover algorithm using O as an oracle'''
    q = QuantumRegister(n)
    qc = QuantumCircuit(q, name="Grover iteration")
    
    
    if (it == 0): it = sqrt(n)

    for _ in range(ceil(it)):
        qc.append(O, q[:])
        
        # # Diffusion Operator
        # # H^n
        # qc.h(q)

        # # Conditional phase shift
        # qc.x(q)

        # qc.h(q[:-1])
        # qc.mct(q[:-1], q[-1])
        # qc.h(q[:-1])
        
        # qc.x(q)

        # # H^n
        # qc.h(q)
        # qc.barrier()
        qc.append(diffuser(n), q[:])
        print(O)
    print(qc)
    return qc
    

def diffuser(nqubits):
    qc = QuantumCircuit(nqubits)
    # Apply transformation |s> -> |00..0> (H-gates)
    for qubit in range(nqubits):
        qc.h(qubit)
    # Apply transformation |00..0> -> |11..1> (X-gates)
    for qubit in range(nqubits):
        qc.x(qubit)
    # Do multi-controlled-Z gate
    qc.h(nqubits-1)
    qc.mct(list(range(nqubits-1)), nqubits-1)  # multi-controlled-toffoli
    qc.h(nqubits-1)
    # Apply transformation |11..1> -> |00..0>
    for qubit in range(nqubits):
        qc.x(qubit)
    # Apply transformation |00..0> -> |s>
    for qubit in range(nqubits):
        qc.h(qubit)
    # We will return the diffuser as a gate
    U_s = qc.to_gate()
    U_s.name = "$U_s$"
    return U_s