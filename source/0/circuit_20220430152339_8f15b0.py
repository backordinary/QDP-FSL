# https://github.com/Amai-RusseMitsu/discrete-time-crystal-ch/blob/127689538c88aad9d20035f78cf86581e3c546bf/.history/setup/circuit_20220430152339.py
import numpy as np
from qiskit import QuantumCircuit

def prepare(circuit,N):
    for i in range(N):
        circuit.h(i)
        circuit.s(i)

def measure(circuit,N):
    for i in range(N):
        circuit.sdg(i)
        circuit.h(i)
        circuit.measure(i,i)

def hamiltonian(circuit,N,t,dt,lamb,J,h,omega):
    
    def coeff(t,h,omega):
        return -h*np.cos(omega*t/2)**2

    #H0 terms with sigma^y and sigma^z
    for i in range(N):
        circuit.ry(2*lamb*dt,i)
        circuit.rz(2*lamb*dt,i)

    circuit.barrier()
    #H1 terms with sigma^z
    for i in range(0,N-1,2):
        circuit.cx(i,i+1)
        circuit.rz(-2*J*dt,i+1)
        circuit.cx(i,i+1)

    for i in range(1,N-1,2):
        circuit.cx(i,i+1)
        circuit.rz(-2*J*dt,i+1)
        circuit.cx(i,i+1)

    circuit.barrier()
    #H2 time dependent terms with sigma^x
    for i in range(N):
        circuit.rx(2*coeff(t,h,omega)*dt,i)