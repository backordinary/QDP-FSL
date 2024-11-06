# https://github.com/jamesjcai/quantum/blob/b18436492456e62db64a8b725554e07068ef2f3e/three_genes_optim.py
from scipy.optimize import minimize

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, Aer, execute
from numpy import pi
from numpy.random import rand
import numpy as np
from numpy.linalg import norm 


N=3






def variational_circuit(qc, qr, theta):
    qc.cry(theta[0], qr[0], qr[1])
    qc.cry(theta[1], qr[1], qr[2])
    qc.cry(theta[2], qr[2], qr[0])
    return qc

def appendkey(dict,key,a):
    if key in dict.keys():
        a.append(dict[key])
    else:
        a.append(0)
    return a



def objective(x):
    p0=[0.1209, 0.0722, 0.0786, 0.1279, 0.1666, 0.0221, 0.4066, 0.0051]
    qreg_q = QuantumRegister(N, 'q')
    creg_c = ClassicalRegister(N, 'c')
    circuit = QuantumCircuit(qreg_q, creg_c)
    for i in range(N):
        circuit.h(qreg_q[i])
    circuit=variational_circuit(circuit,qreg_q,x)    
    circuit.measure(qreg_q[0], creg_c[0])
    circuit.measure(qreg_q[1], creg_c[1])
    circuit.measure(qreg_q[2], creg_c[2])
    shots=10000
    backend = Aer.get_backend('qasm_simulator')
    job=execute(circuit, backend, shots=shots, memory=False)
    result=job.result()
    counts=result.get_counts(circuit)
    #a=[]
    #b=[]
    #for id, key in enumerate(counts):
    #    a.append(key)
    #    b.append(counts[key])
    #c=np.array(b)[np.argsort(a).astype(int)]
    c=[]
    c=appendkey(counts,'000',c)
    c=appendkey(counts,'001',c)
    c=appendkey(counts,'010',c)
    c=appendkey(counts,'011',c)
    c=appendkey(counts,'100',c)
    c=appendkey(counts,'101',c)
    c=appendkey(counts,'110',c)
    c=appendkey(counts,'111',c)
    c=np.array(c)
    p=c / sum(c)
    print(norm(p-p0))
    return norm(p-p0)


r_min, r_max = 0, 2*pi
theta = r_min + rand(N) * (r_max - r_min)
bnds = tuple((0,2*pi) for x in theta)

print(theta)
# result = minimize(objective, theta, method='L-BFGS-B')
result = minimize(objective, theta, method='Powell', bounds=bnds)
# result = minimize(objective, theta, method='SLSQP', bounds=bnds, tol=0.001,options={'verbose': 1})

# https://stackoverflow.com/questions/18767657/how-do-i-use-a-minimization-function-in-scipy-with-constraints

# [0.1209 0.0722 0.0786 0.1279 0.1666 0.0221 0.4066 0.0051]
# [0.46190135 2.79787652 4.30121555]
result

