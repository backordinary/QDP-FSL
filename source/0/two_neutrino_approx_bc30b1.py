# https://github.com/pablocastano03/quantum_computing_and_neutrinos/blob/39e37718fe5b0975b3d30a6aa25c290eafb34e66/two_neutrino_approx.py
from qiskit import Aer, execute, ClassicalRegister, QuantumRegister, QuantumCircuit

import numpy as np

from physicsconstants import OscParams

def two_nu_phi(L,E):
    r'''
    '''
    coef = 1.88e-3
    #coef = 1.88e-2
    phi  = coef*L/E
    return phi

def two_nu_qc(theta, phi, draw=True):
    q = QuantumRegister(1)
    c = ClassicalRegister(1)
    qc = QuantumCircuit(q,c)
    qc.x(q)
    qc.u(-2*theta,0,0,q) # PMNS
    qc.rz(phi, q)        # Time-evolution
    qc.u(2*theta,0,0,q)  # PMNS dagger
    qc.measure(q,c)
    if draw:
        qc.draw()
    return qc

if __name__=='__main__':
    E = 1 # GeV
    LL = np.linspace(0, 1e4, 100) # km
    op = OscParams()
    
    n = 100000
    results = np.zeros((2,len(LL)))

    for i, L in enumerate(LL):
        draw = (i==0)
        phi  = two_nu_phi(L, E)
        qc   = two_nu_qc(op.theta12, phi, draw=draw)
        job  = execute(qc,Aer.get_backend('qasm_simulator'),shots=n)
        counts = job.result().get_counts(qc)
        for j, (key, val) in enumerate(sorted(counts.items())):
            results[j, i] = float(val)/n
    np.save('two_neutrino_approx', results)
