# https://github.com/Diegoj029/Stranger-Ducks/blob/65e41b9e7d09927e2d1f0fe5e09ed78d65616ccd/Qtools.py
import numpy as np
import random
from qiskit import QuantumCircuit, Aer, execute
from scipy.spatial.distance import cosine

def measuring(circuit, backend, shots=2000):
    job = execute(circuit, backend, shots=shots)
    result = job.result()
    counts = result.get_counts()
    return {key: value*100.0/shots for key, value in counts.items()}

def add_gate(circuit, gate=None, channel=0, channel_op=0):
    if gate == "hadamard":
        circuit.h(channel)
    
    elif gate == "x":
        circuit.x(channel)
    
    elif gate == "y":
        circuit.y(channel)
    
    elif gate == "z":
        circuit.z(channel)
    
    elif gate == "cnot":
        circuit.cnot(channel, channel_op)
        
    return circuit

def randomQuantumState():
    qc = QuantumCircuit(3,3)
    for i in range(0,int(random.randint(1,6))):
        g = random.choice(("hadamard","x","y","z","cnot"))
        if g == "cnot":
            q = random.sample(tuple(range(0,3)),2)
            add_gate(qc,gate=g,channel=q[0],channel_op=q[1])
        else:
            q = random.choice(tuple(range(0,3)))
            add_gate(qc,gate=g,channel=q)

    qc.barrier()
    qc.measure([0,1,2],[0,1,2])
    
    simulator = Aer.get_backend('qasm_simulator')
    rqs = measuring(qc, backend=simulator)
    return rqs

def dictovec(dic):
    base_states = [f"{a}"+f"{b}"+f"{c}" for a in range(2) for b in range(2) for c in range(2)]
    
    for base in base_states:
        if base in dic.keys():
            continue
        else:
            dic[base] = 0.0
        
    return np.array([dic[base] for base in base_states])


def similarity(dic1, dic2, err=0.01):
    v1 = dictovec(dic1)
    v2 = dictovec(dic2)
    distance = cosine(v1, v2)
    
    if distance <= err:
        return True, distance
    else:
        return False, distance
    
def inicioRandom():
    base_states = [f"{a}"+f"{b}"+f"{c}" for a in range(2) for b in range(2) for c in range(2)]
    base = random.choice(base_states)
    qc = QuantumCircuit(3,3)
    if base[0] == "1":
        qc.x(0)
    if base[1] == "1":
        qc.x(1)
    if base[2] == "1":
        qc.x(2)
    base = "|" + base + ">"
    return qc,base

def dictostr(dic):
    str_orden = ""
    for key, value in dic.items():
        if value >= 5.0:
            str_orden += f"|{key}>: {value}% ; "
        
    return str_orden
