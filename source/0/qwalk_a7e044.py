# https://github.com/kgeorgopoulos2/programbench/blob/d083b3e7f500dc599b53effdf02c5cc9cf9070fd/qwalk.py
# importing QISKit
from qiskit import Aer, IBMQ
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, execute

# Toffoli gate for n qubits
def ncx(circ, tgt, ctrl, anc):
    '''Create a generalised CNOT gate with more than 2 control qubits.'''
    # Compute
    circ.ccx(ctrl[0], ctrl[1], anc[0])
    for i in range(2, len(ctrl)):
        circ.ccx(ctrl[i], anc[i-2], anc[i-1])
    
    # CNOT
    circ.cx(anc[len(ctrl)-2], tgt)

    # Uncompute
    for i in range(len(ctrl)-1, 1, -1):
        circ.ccx(ctrl[i], anc[i-2], anc[i-1])
    circ.ccx(ctrl[0], ctrl[1], anc[0])
    
    return circ

# Define increment function   
def inc(q_circuit, q_reg, q_anc, q_coin, n):
    '''Increment circuit for n-qubit states. Specifficaly, n=logN.''' 
    for i in range(n, 2, -1):
        tgt = q_reg[i-1] # Target qubit is the MSQ
        ctrl = []
        for j in range(0, i-1):
            ctrl.append(q_reg[j])
        ctrl.append(q_coin[0])
        q_circuit = ncx(q_circuit, tgt, ctrl, q_anc)
    
    q_circuit.ccx(q_coin[0], q_reg[0], q_reg[1])
    q_circuit.cx(q_coin[0], q_reg[0])
    
    return q_circuit

# Define decrement function
def dec(q_circuit, q_reg, q_anc, q_coin, n):
    '''Decrement circuit for n-qubit states. Specifficaly, n=logN.'''
    q_circuit.x(q_coin[0]) # Reverse the coin to decrease

    # Circuit
    for i in range(0, n-1):
        q_circuit.cx(q_coin[0], q_reg[i]) # Invert the registers necessary

    for i in range(n, 2, -1):
        tgt = q_reg[i-1]
        ctrl = []
        for j in range(0, i-1):
            ctrl.append(q_reg[j])
        ctrl.append(q_coin[0])
        q_circuit = ncx(q_circuit, tgt, ctrl, q_anc)
    
    q_circuit.ccx(q_coin[0], q_reg[0], q_reg[1])
    q_circuit.cx(q_coin[0], q_reg[0])

    for i in range(0, n-1):
        q_circuit.cx(q_coin[0], q_reg[i]) # Uncompute the inverted registers

    q_circuit.x(q_coin[0]) # Uncompute the reversion of the coin
    
    return q_circuit

# Function that generates the quantum circuit. Also returns the number of gates
# within the quantum circuit
def walk(n, t):
    '''Function that generates the quantum circuit for the quantum walk. Arguments are given
    as the number of qubits needed to describe the state space, n, and the time steps (i.e coin 
    flips or iterations of the walk), t.'''
    if (n > 2):
        # Create the Quantum Circuit
        q_reg = QuantumRegister(n, 'q')
        q_coin = QuantumRegister(1, 'coin')
        q_anc = QuantumRegister(n-1, 'anc')
        c_reg = ClassicalRegister(n, 'c')
        qwalk_circuit = QuantumCircuit(q_reg, q_anc, q_coin, c_reg)

        def runQWC(qwalk_circuit, steps):
            for i in range(steps):
                qwalk_circuit.h(q_coin[0])
                qwalk_circuit = inc(qwalk_circuit, q_reg, q_anc, q_coin, n)
                qwalk_circuit = dec(qwalk_circuit, q_reg, q_anc, q_coin, n)

            qwalk_circuit.measure(q_reg, c_reg)

            return qwalk_circuit

        steps = t
        qwalk_circuit = runQWC(qwalk_circuit, steps)

    else:
        q_reg = QuantumRegister(n, 'q')
        q_coin = QuantumRegister(1, 'coin')
        c_reg = ClassicalRegister(n, 'c')
        qwalk_circuit = QuantumCircuit(q_reg, q_coin, c_reg)
        
        def runQWC(qwalk_circuit, steps):
            for i in range(steps):
                qwalk_circuit.h(q_coin[0])
                qwalk_circuit.ccx(q_coin[0], q_reg[0], q_reg[1])
                qwalk_circuit.cx(q_coin[0], q_reg[0])
                qwalk_circuit.x(q_coin[0])
                qwalk_circuit.cx(q_coin[0], q_reg[0])
                qwalk_circuit.ccx(q_coin[0], q_reg[0], q_reg[1])
                qwalk_circuit.cx(q_coin[0], q_reg[0])
                qwalk_circuit.cx(q_coin[0], q_reg[0])
                qwalk_circuit.x(q_coin[0])

            qwalk_circuit.measure(q_reg, c_reg)

            return qwalk_circuit

        steps = t
        qwalk_circuit = runQWC(qwalk_circuit, steps)
    
    return qwalk_circuit