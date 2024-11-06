# https://github.com/was-albi/QAOA-CellFreeMIMO/blob/2ed574f8f6334f8d3ecb59873e4d23f4df898067/src/QAOA_utilities.py
import numpy as np
import scipy.special

# Importing standard Qiskit libraries
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, transpile, Aer, IBMQ, execute
from qiskit.circuit import Parameter
from qiskit.providers.aer import QasmSimulator
from qiskit.circuit.library.standard_gates import RYGate, XXPlusYYGate


def SCS(n,l):
    # Split & Cyclic Shift circuits
    qc = QuantumCircuit(n)

    qc.cnot(l-2,l-1)
    qc.cry(2*np.arccos(np.sqrt(1/l)),l-1,l-2)
    qc.cnot(l-2,l-1)

    for i in range(l-1,1,-1):
        qc.cnot(i-2,l-1)
        #print((l-i+1)/l)
        CCRY=RYGate(2*np.arccos(np.sqrt((l-i+1)/l))).control(2)
        qc.append(CCRY,(l-1,i-1,i-2))
        qc.cnot(i-2,l-1)

    return(qc)


def Unn(n):
    # Create a circuit that is able to produce Dicke states.
    # See A. Bärtschi and S. Eidenbenz, "Deterministic Preparation of Dicke States" (2019)
    Unn = QuantumCircuit(n)

    for l in range(n,1,-1):
        sc = SCS(n,l)
        Unn = Unn + sc

    return(Unn)

def U_alpha(n, m):
    # Create U_alpha unitary
    # n = MK, m = M^
    qc = QuantumCircuit(n)
    qc.x(n-1)
    residual = 0
    BW_card = 0
    for i in range(m):
        BW_card += scipy.special.comb(n,i+1)

    for i in range(m-1):
        numerator = scipy.special.comb(n, i+1)
        qc.cry(2*np.arccos(np.sqrt( numerator/( BW_card - residual ) )), n-(i+1),n-(i+2))
        residual += numerator

    return(qc)


def Um_ring(n,beta):
    qc = QuantumCircuit(n)
    XXYY = XXPlusYYGate(4*beta, beta=0)
    i = 0
    while i < n-1:
        qc.append(XXYY, (i,i+1))
        i += 2
    i = 1
    while i < n-1:
        qc.append(XXYY, (i,i+1))
        i += 2

    if n % 2 != 0:
        qc.append(XXYY, (n-1,0))

        return(qc)
