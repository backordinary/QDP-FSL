# https://github.com/ugo-nzongani/TER/blob/a5727314248af8fd7e0002a34dc5e4576e947b77/fujiwara_circuit.py
# -*- coding: utf-8 -*-
from qiskit import *
from qiskit.extensions import *
from math import pow, sqrt
import numpy as np
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram
from qiskit.tools.monitor import job_monitor
# %matplotlib inline
# %config InlineBackend.figure_format = 'svg'

# Implementation of Fujiwara's circuit with a position-dependent coin operator

# In Qiskit the highest qubit is the least significant one

'''
Input :
    n : number of qubits encoding the position
Outputs : Z+ and Z- gates used to move the walker to the right or to the left
'''
def get_z_gates(n):
    pos = QuantumRegister(n, name="pos")
    coin = QuantumRegister(1, name="coin")
    qc = QuantumCircuit(pos, coin)
    for i in range(n):
        ctrl = [coin[0]] + pos[:n-i-1]
        qc.mct(ctrl, [pos[n-i-1]])
    z_up = qc.to_gate(label="Z+")
    z_down = qc.inverse().to_gate(label="Z-")
    return z_up, z_down

'''
Inputs :
    qc : quantum circuit
    pos : position register
    coin : coin register
    n : number of qubits encoding the position
    z_up : Z+ gate
    z_down : Z- gate
    
    Optionals arguments :
        position_dependent : True if we want a position-dependent coin operator
        coin_list : In case we want a position-dependent coin operator, coin_list[i] contains the coin operator
                    we want to apply on position i
Output : Add a walk step to the circuit
'''
def step(qc, pos, coin, n, z_up, z_down, position_dependent=False, coin_list=[]):
    if(position_dependent):
        qubits = [pos[i] for i in range(n)]
        for i in range(len(coin)):
            qubits.append(coin[i])
        qc_position_dependent = get_position_dependent_gate(n, coin_list)
        gate = qc_position_dependent.to_gate(label="C")
        qc.append(gate, qubits)
    else:
        qc.h(coin[0])
    # the walk
    qc.barrier()
    # Z+
    qubits = [pos[i] for i in range(n)] + [i for i in coin]
    qc.append(z_up, qubits)
    # NOT GATE
    qc.x(coin[0])
    # Z-
    qc.append(z_down, qubits)
    # NOT GATE
    qc.x(coin[0])

'''
Inputs:
    n : number of qubits encoding the position
    n_step : number of steps
    
    Optionals arguments :
        position_dependent : True if we want a position-dependent coin operator
        coin_list : In case we want a position-dependent coin operator, coin_list[i] contains the coin operator
                    we want to apply on position i
Output : quantum circuit to compute the walk
'''
def circuit(n, n_step, position_dependent=False, coin_list=[]):
    # Position register
    pos = QuantumRegister(n, name="pos")
    # Coin register
    coin = QuantumRegister(1, name="coin")
    # Classical register used to store the measurements outcomes
    c = ClassicalRegister(n, name="res")
    qc = QuantumCircuit(pos, coin, c)
    z_up, z_down = get_z_gates(n)
    
    qc.barrier()
    # steps
    for i in range(n_step):
        step(qc, pos, coin, n, z_up, z_down, position_dependent, coin_list)
        qc.barrier()
    # we only measure the position register
    qc.measure(pos,c)
    return qc

'''
Inputs:
    n : number of qubits encoding the position
    coin_list : coin_list[i] contains the coin operator we want to apply on position i
Output : circuit implementing the position-dependent coin operator
'''
def get_position_dependent_gate(n, coin_list):
    # Position register
    pos = QuantumRegister(n, name="pos")
    # Coin register
    coin = QuantumRegister(1, name="coin")
    qc = QuantumCircuit(pos, coin)
    # qubits on which the coin operator acts
    gate_qubits = [coin[k] for k in range(len(coin))]
    # all the qubits encoding the position are getting controlled
    ctrl_qubits = [pos[k] for k in range(n)]
    qubits = ctrl_qubits + gate_qubits
    for i in range(int(pow(2,n))):
        binary = bin(i)[2:].zfill(n)
        for j in range(n):
            if i % int(pow(2,n-j-1)) == 0:
                qc.x(pos[n-j-1])
        gate = UnitaryGate(coin_list[int(binary,2)], label=binary)
        qc.append(gate.control(n), qubits)
    return qc

'''
Inputs :
    qc : quantum circuit
    n_shots : number of iteration
Output : the measurement outcomes obtained with a simulation
'''
def execute_simulation(qc, n_shots):
    '''
    backend = BasicAer.get_backend('qasm_simulator')
    #job = execute(qc, backend, shots=n_shots)
    #counts = dict(job.result().get_counts(qc))
    job = execute(qc, backend=backend, shots=n_shots)
    counts = dict(job.result().get_counts(qc))
    '''
    backend_sim = Aer.get_backend('qasm_simulator')
    job_sim = backend_sim.run(transpile(qc, backend_sim), shots=n_shots)
    result_sim = job_sim.result()
    counts = result_sim.get_counts(qc)
    return counts

'''
EXAMPLE with no position-dependent coin operator

n = 3 # number of qubits encoding the position
n_step = 2 # number of steps
circ1 = circuit(n, n_step)
res = execute_simulation(circ1, 1024)
plot_histogram(res)
'''

'''
EXAMPLE with position-dependent coin operator

coin_list = []
Hadamard = np.array([[1,1],[1,-1]])/np.sqrt(2)
Z = np.array([[1,0],[0,-1]])
for i in range(int(pow(2,n))):
    if i % 2 == 0:
        coin_list.append(Hadamard)
    else:
        coin_list.append(Z)
        
circ2 = circuit(n, n_step, position_dependent=True, coin_list=coin_list)
'''
