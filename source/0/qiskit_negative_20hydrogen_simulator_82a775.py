# https://github.com/Agdanpanda/QiskitEssaySupplement/blob/e11b86bebee90acd07c134f4dae524374b006813/QISKIT_negative%20hydrogen_Simulator.py
import numpy as np
# Importing standard Qiskit libraries
from qiskit import QuantumCircuit, transpile, Aer, IBMQ
from qiskit import execute
from qiskit.tools.jupyter import *
from qiskit.visualization import *
#from ibm_quantum_widgets import *
from qiskit.exceptions import QiskitError
from qiskit import ClassicalRegister, QuantumRegister
from qiskit.circuit import quantumcircuit
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.providers.ibmq import least_busy
from scipy.optimize import minimize
import matplotlib.pyplot as mpl

def Z1(theta):
    q = QuantumRegister(2)
    c = ClassicalRegister(2)
    qc = QuantumCircuit(q,c)
    qc.u1(theta[0] , q[0])
    qc.u3(theta[1], -np.pi/2, np.pi/2, q[0])
    qc.u1(theta[2], q[0])
    qc.cx(q[0],q[1])
    qc.u1(theta[3], q[1])
    qc.u3(theta[4], -np.pi/2, np.pi/2, q[1])
    qc.u1(theta[5],q[1])
    qc.cx(q[1], q[0])
    qc.u1(theta[6],q[0])
    qc.u1(theta[7],q[1])
    qc.u3(theta[8], -np.pi/2, np.pi/2, q[0])
    qc.u3(theta[9], -np.pi/2, np.pi/2, q[1])
    qc.u1(theta[10], q[0])
    qc.u1(theta[11], q[1])
    qc.z(q[0])
    qc.measure(q[0], c[0])
    qc.measure(q[1], c[1])

    shots = T
    max_credits = 3
    
    job_hpc = execute(qc, backend,  shots = shots, max_credits = max_credits )
    try:        
        result_hpc = job_hpc.result()
    except:
        print('Error found: ',job_hpc.error_message() )
    counts12 = result_hpc.get_counts(qc)
    Z = 0 
    if '00' in list(counts12):
        Z = Z + counts12['00']/T
    if '01' in list(counts12):
        Z = Z + counts12['01']/T
    if '10' in list(counts12):
        Z = Z - counts12['10']/T
    if '11' in list(counts12):
        Z = Z - counts12['11']/T  
    return Z
def Z2(theta):
    q = QuantumRegister(2)
    c = ClassicalRegister(2)
    qc = QuantumCircuit(q,c)
    qc.u1(theta[0] , q[0])
    qc.u3(theta[1], -np.pi/2, np.pi/2, q[0])
    qc.u1(theta[2], q[0])
    qc.cx(q[0],q[1])
    qc.u1(theta[3], q[1])
    qc.u3(theta[4], -np.pi/2, np.pi/2, q[1])
    qc.u1(theta[5],q[1])
    qc.cx(q[1], q[0])
    qc.u1(theta[6],q[0])
    qc.u1(theta[7],q[1])
    qc.u3(theta[8], -np.pi/2, np.pi/2, q[0])
    qc.u3(theta[9], -np.pi/2, np.pi/2, q[1])
    qc.u1(theta[10], q[0])
    qc.u1(theta[11], q[1])
    qc.z(q[1])
    qc.measure(q[0], c[0])
    qc.measure(q[1], c[1])
    shots = T
    max_credits = 3
    job_hpc = execute(qc, backend,  shots = shots, max_credits = max_credits )
    try:        
        result_hpc = job_hpc.result()
    except:
        print('Error found: ',job_hpc.error_message() )
    counts12 = result_hpc.get_counts(qc)
    Z = 0 
    if '00' in list(counts12):
        Z = Z + counts12['00']/T
    if '01' in list(counts12):
        Z = Z - counts12['01']/T
    if '10' in list(counts12):
        Z = Z + counts12['10']/T
    if '11' in list(counts12):
        Z = Z - counts12['11']/T 
    return Z
def Z3(theta):
    q = QuantumRegister(2)
    c = ClassicalRegister(2)
    qc = QuantumCircuit(q,c)
    qc.u1(theta[0] , q[0])
    qc.u3(theta[1], -np.pi/2, np.pi/2, q[0])
    qc.u1(theta[2], q[0])
    qc.cx(q[0],q[1])
    qc.u1(theta[3], q[1])
    qc.u3(theta[4], -np.pi/2, np.pi/2, q[1])
    qc.u1(theta[5],q[1])
    qc.cx(q[1], q[0])
    qc.u1(theta[6],q[0])
    qc.u1(theta[7],q[1])
    qc.u3(theta[8], -np.pi/2, np.pi/2, q[0])
    qc.u3(theta[9], -np.pi/2, np.pi/2, q[1])
    qc.u1(theta[10], q[0])
    qc.u1(theta[11], q[1])
    qc.z(q[0])
    qc.z(q[1])
    qc.measure(q[0], c[0])
    qc.measure(q[1], c[1])
    shots = T
    max_credits = 3
    job_hpc = execute(qc, backend,  shots = shots, max_credits = max_credits )
    try:        
        result_hpc = job_hpc.result()
    except:
        print('Error found: ',job_hpc.error_message() )
    counts12 = result_hpc.get_counts(qc)
    Z = 0 
    if '00' in list(counts12):
        Z = counts12['00']/T
    if '01' in list(counts12):
        Z = Z - counts12['01']/T
    if '10' in list(counts12):
        Z = Z - counts12['10']/T
    if '11' in list(counts12):
        Z = Z + counts12['11']/T   
    return Z
def totalEnergy(theta):
    z1Val = Z1(theta)
    z2Val = Z2(theta)
    z3Val = Z3(theta)
    h1 = -0.5
    h2 = -0.5
    h3 = 0.625
    term1 = 1/2*h1*( 1.0 - z1Val )
    term2 = 1/2*h2*( 1.0 - z3Val )
    term3 = h3/4*( 1.0 - z1Val + z2Val - z3Val )
    energy = term1 + term2 + term3
    print('term1: ', term1, 'Z1: ', z1Val)
    print('term2: ', term2, 'Z2: ', z2Val)
    print('term3: ', term3, 'Z3: ', z3Val)
    print('Current Energy: ',energy)
    return energy
backend = Aer.get_backend('qasm_simulator')
print('Backend:',backend)
T = 10000
theta0 = [0.2, 2, 0.8, 0, 0.2, -1.1, 1.15, 0, 3, 1, 0.2, 0.7 ]
res = minimize( totalEnergy, theta0, method = 'COBYLA', options = {'tol':1e-6, 'disp': True})
print('totalEnergy: ',res.fun)
print('Theta: ', res.x)
