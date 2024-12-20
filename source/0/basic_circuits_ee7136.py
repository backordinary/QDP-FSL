# https://github.com/andrewcpotter/holopy/blob/f0439265826d3c9a89a00a87f7cc394ed095ef93/circuits/basic_circuits.py
"""
Predefined (parameterized) qiskit circuits for holovqe

Created: 2/14/2021, AC Potter

Modified: 3/22/2021, YX Zhang
"""
# standard imports
import sys
sys.path.append("..") # import one subdirectory up in files
import numpy as np
import qiskit as qk 

# holopy imports
from networks.isonetwork import QKParamCircuit

#%% Two-qubit circuits
def add_1q_circ(circ,q,params):
    """
    general 1 qubit gate
    inputs:
        - q1,2 qubits
        - params, qiskit ParameterVector object or list of qk Parameters
    returns: 
        - QKParamCircuit object
    """
    
    # 1q gates
    # physical qubit
    circ.rx(params[0],q)
    circ.rz(params[1],q)
    circ.rx(params[2],q)

def add_su4_circ(circ,q1,q2,params):
    """
    inputs:
        - q1,2 qubits
        - params, qiskit ParameterVector object or list of qk Parameters
    returns: 
        - QKParamCircuit object
    """
    
    # 1q gates
    # physical qubit
    circ.rx(params[0],q1)
    circ.rz(params[1],q1)
    circ.rx(params[2],q1)
    # bond qubit
    circ.rx(params[3],q2)
    circ.rz(params[4],q2)
    circ.rx(params[5],q2)
    
    # two qubit gates
    # xx-rotation
    [circ.h(q) for q in [q1,q2]]
    circ.cx(q1,q2)
    circ.rz(params[6],q2)
    circ.cx(q1,q2)
    [circ.h(q) for q in [q1,q2]]
    # yy-rotation
    [circ.rx(np.pi/2,q) for q in [q1,q2]]
    circ.cx(q1,q2)
    circ.rz(params[7],q2)
    circ.cx(q1,q2)
    [circ.rx(-np.pi/2,q) for q in [q1,q2]]
    # zz-rotation
    circ.cx(q1,q2)
    circ.rz(params[8],q2)
    circ.cx(q1,q2)
    
    # 1q gates
    # physical qubit
    circ.rx(params[9],q1)
    circ.rz(params[10],q1)
    circ.rx(params[11],q1)
    # bond qubit
    circ.rx(params[12],q2)
    circ.rz(params[13],q2)
    circ.rx(params[14],q2)

def add_u4_circ(circ,q1,q2,params):
    circ.u3(params[0],params[1],params[2],q1)
    circ.u3(params[3],params[4],params[5],q2)
    circ.cx(q2,q1)
    circ.rz(params[6],q1)
    circ.ry(params[7],q2)
    circ.rz(params[8],q2)
    circ.cx(q1,q2)
    circ.ry(params[9],q2)
    circ.cx(q2,q1)
    circ.u3(params[10],params[11],params[12],q1)
    circ.u3(params[13],params[14],params[15],q2)
    

def add_xx_gate(circ,q1,q2,param):
    [circ.h(q) for q in [q1,q2]]
    circ.cx(q1,q2)
    circ.rz(param,q2)
    circ.cx(q1,q2)
    [circ.h(q) for q in [q1,q2]]    

def add_xx_circ(circ,q1,q2,params):
    """
    inputs:
        - q1,2 qubits
        - params, qiskit ParameterVector object or list of qk Parameters
    returns: 
        - QKParamCircuit object
    """
    
    # two qubit gates
    # xx+yy-rotation
    circ.rx(np.pi/2,q1)
    circ.ry(np.pi/2,q2)
    circ.cz(q1,q2)
    circ.rx(-params[0],q1)
    circ.ry(params[0],q2)
    circ.cz(q1,q2)
    circ.rx(-np.pi/2,q1)
    circ.ry(-np.pi/2,q2)

def add_xxz_circ(circ,q1,q2,params):
    """
    inputs:
        - q1,2 qubits
        - params, qiskit ParameterVector object or list of qk Parameters
    returns: 
        - QKParamCircuit object
    """
    
    # two qubit gates
    # xx+yy-rotation
    circ.rx(np.pi/2,q1)
    circ.ry(np.pi/2,q2)
    circ.cz(q1,q2)
    circ.rx(-params[0],q1)
    circ.ry(params[0],q2)
    circ.cz(q1,q2)
    circ.rx(-np.pi/2,q1)
    circ.ry(-np.pi/2,q2)
    # zz-rotation
    circ.cx(q1,q2)
    circ.rz(params[1],q2)
    circ.cx(q1,q2)

def add_xyz_circ(circ,q1,q2,params):
    """
    inputs:
        - q1,2 qubits
        - params, qiskit ParameterVector object or list of qk Parameters
    returns: 
        - QKParamCircuit object
    """
    
    # two qubit gates
    # xx-rotation
    [circ.h(q) for q in [q1,q2]]
    circ.cx(q1,q2)
    circ.rz(params[0],q2)
    circ.cx(q1,q2)
    [circ.h(q) for q in [q1,q2]]
    # yy-rotation
    [circ.rx(np.pi/2,q) for q in [q1,q2]]
    circ.cx(q1,q2)
    circ.rz(params[1],q2)
    circ.cx(q1,q2)
    [circ.rx(-np.pi/2,q) for q in [q1,q2]]
    # zz-rotation
    circ.cx(q1,q2)
    circ.rz(params[2],q2)
    circ.cx(q1,q2)
    
def add_ising_circ(circ,q1,q2,params,include_end_rx2=False):
    """
    inputs:
        - q1,2 qubits
        - params, qiskit ParameterVector object or list of qk Parameters
    returns: 
        - QKParamCircuit object
    """
    # 1q gates
    circ.rx(params[0],q1) # physical qubit
    circ.rx(params[1],q2) # bond qubit
    
    # cartan block 
    # xx-rotation
    [circ.h(q) for q in [q1,q2]]
    circ.cx(q1,q2)
    circ.rz(params[2],q2)
    circ.cx(q1,q2)
    [circ.h(q) for q in [q1,q2]]
    # yy-rotation
    [circ.rx(np.pi/2,q) for q in [q1,q2]]
    circ.cx(q1,q2)
    circ.rz(params[3],q2)
    circ.cx(q1,q2)
    [circ.rx(-np.pi/2,q) for q in [q1,q2]]
    # zz-rotation
    circ.cx(q1,q2)
    circ.rz(params[4],q2)
    circ.cx(q1,q2)
    
    # 1q gates
    circ.rx(params[5],q1) # physical qubit
    if include_end_rx2: circ.rx(params[1],q2) # bond qubit
    
    
    

#%% Multi-qubit circuits
def star_circ(qp,qb,label,circ_type='su4'):
    """
    sequentially interacts qp with each q in qb
    inputs:
        - qp, quantum register w/ 1 physical qubit
        - qb, quantum register w/ N bond qubits
        - label, str, label for circuit
        - circ_type, str, 'su4','xxz',etc...
    outputs:
        - parameterized circuit
        - list of parameters
    """
    nb = len(qb) # number of bond qubits
    circ = qk.QuantumCircuit(qp,qb)
    
    # parse number of parameters
    if circ_type=='su4': 
        param_per_circ = 15
        n_params = param_per_circ*nb
        circ_fn = add_su4_circ
        
    elif circ_type=='xxz':
        param_per_circ = 3
        n_params = param_per_circ*nb
        circ_fn = add_xxz_circ
    else:
        raise NotImplementedError(circ_type+' not implemented')
        
    params = [qk.circuit.Parameter(label+str(j)) for j in range(n_params)]#qk.circuit.ParameterVector(label,length=n_params)
    for i in range(nb):
        circ_fn(circ,qp[0],
                qb[i],
                params[param_per_circ*i:param_per_circ*i+param_per_circ])
    
    param_circ = QKParamCircuit(circ,params)
    return param_circ,params
        


#%% test/debug
# qp = qk.QuantumRegister(1) # physical qubit
# qb = qk.QuantumRegister(2) # bond qubit 
# label = 'c1' # label for circuit
# circ,params = star_circ(qp, qb, label)
# circ.circ.draw('mpl',scale=0.4)
