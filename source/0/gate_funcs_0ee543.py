# https://github.com/kemperlab/Cartan_DMFT/blob/e80c74c2d250af37523f9fa5508246ed016ce89b/src/functions/gate_funcs.py
"""
Script and Functions to assmeble gates for the DMFT Loop
"""
from CQS.util.PauliOps import I
from CQS.util.verification import Nident
import qiskit
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, execute
from collections import OrderedDict
from openfermion.ops import FermionOperator, QubitOperator
from openfermion.utils import hermitian_conjugated, commutator

from qiskit import QuantumCircuit, Aer
#import pytket as pytk
import numpy as np
from CQS.util.IO import tuplesToMatrix
from CQS.util.verification import Nident
from scipy import optimize
from numpy import kron


def CQS2OFterms(opList,coList):
    idx2pauli = ['I','X','Y','Z']
    OFterms = OrderedDict()
    for (tm, co) in zip(opList, coList):
        op = tuple((ii,idx2pauli[jj]) for ii,jj in enumerate(tm)  if jj)
        OFterms[op] = co
    return OFterms

    FO = FermionOperator

def OpenFermionToCQS(H, systemSize):
        """
        Converts the Operators to a list of (PauliStrings)

        Args:
            H(obj): The OpenFermion Operator
            systemSize (int): The number of qubits in the system
        returns (tuple): (coefficient, pauliString)
        """
        stringToTuple  = {   
                        'X': 1,
                        'Y': 2,
                        'Z': 3
                    }
        opList = []
        coList = [] 
        for op in H.terms.keys(): #Pulls the operator out of the QubitOperator format
            coList.append(H.terms[op])
            opIndexList = []
            opTypeDict = {}
            tempTuple = ()
            for (opIndex, opType) in op:
                opIndexList.append(opIndex)
                opTypeDict[opIndex] = opType
            for index in range(systemSize):
                if index in opIndexList:
                    tempTuple += (stringToTuple[opTypeDict[index]],)
                else:
                    tempTuple += (0,)
            opList.append(tempTuple)
        return (coList, opList)

QO = QubitOperator



# define quantum circuit gates with OpenFermion sparse string format,
# then the full circuit unitary can be simplified with OpenFermion sparse string 
# algorithm without using numerical matrices
class OFgate:
    def __init__(self, nqubit=1):
        self.nq = nqubit
    
    def s(self, q):
        return  0.5*(QO(((q,'Z'))) + QO(())) - 0.5j*(QO(((q,'Z'))) - QO(()))
    
    def h(self, q):
        return (QO(((q,'X'))) + QO(((q,'Z'))))/np.sqrt(2)
    
    def cx(self, c,t):
        return 0.5*(QO(((c,'Z'))) + QO(())) - 0.5*(QO(((c,'Z'))) - QO(()))*QO(((t,'X')))
    
    def cz(self, c,t):
        return 0.5*(QO(((c,'Z'))) + QO(())) - 0.5*(QO(((c,'Z'))) - QO(()))*QO(((t,'Z')))   

def getGroundStateCircuitFunc(HamiltonianObj):
    """
    Computes the ground state of the system via simulated VQE and prepares the circuit

    Temporary: Prepares the state using faked VQE
    """
    #Gets the exact matrix of the Hamiltonian
    Hamiltonian = 0
    for (co, p) in zip(HamiltonianObj.HCoefs, HamiltonianObj.HTuples):
        #p = p[::-1]
        Hamiltonian = Hamiltonian + tuplesToMatrix(co, p)
    #Defines the function to minimize over, the inner product
    def Psi(theta):
        vqeCirc = QuantumCircuit(4,4)

        vqeCirc.ry(theta[0].real,0)
        vqeCirc.ry(theta[1].real,1)
        vqeCirc.ry(theta[2].real,2)
        vqeCirc.ry(theta[3].real,3)
        vqeCirc.cnot(2,3)
        vqeCirc.ry(theta[4].real,2)
        vqeCirc.ry(theta[5].real,3)
        vqeCirc.cnot(0,2)
        vqeCirc.ry(theta[6].real,0)
        vqeCirc.ry(theta[7].real,2)
        vqeCirc.cnot(0,1)

        backend = Aer.get_backend('statevector_simulator')
        result = qiskit.execute(vqeCirc, backend=backend).result()
        psi0 =  result.get_statevector(vqeCirc)
        return (np.conjugate(psi0.T) @ Hamiltonian @ psi0).real
    #Minimizes over the VQE function
    minObj = optimize.minimize(Psi, [1, 1, 1, 1, 1, 1, 1, 1], options={'disp':True},tol=1e-8)
    params = minObj.x
    #Buidls the circuit that we actually wanted
    def GroundStateCircuit(circuit):
        circuit.ry(params[0].real,0)
        circuit.ry(params[1].real,1)
        circuit.ry(params[2].real,2)
        circuit.ry(params[3].real,3)
        circuit.cnot(2,3)
        circuit.ry(params[4].real,2)
        circuit.ry(params[5].real,3)
        circuit.cnot(0,2)
        circuit.ry(params[6].real,0)
        circuit.ry(params[7].real,2)
        circuit.cnot(0,1)
        return circuit

    return GroundStateCircuit

def getGroundStateShortCircuitFunc(HamiltonianObj):
    """
    Prepare the ground state using VQE type circuit
    """
    #Gets the exact matrix of the Hamiltonian
    Hamiltonian = 0  #np.zeros(2**4,dtype=complex)
    for (co, p) in zip(HamiltonianObj.HCoefs, HamiltonianObj.HTuples):
        reverseP = p[::-1]
        Hamiltonian = Hamiltonian + tuplesToMatrix(co, reverseP)
    #Defines the function to minimize over, the inner product
    def Psi(theta):
        vqeCirc = QuantumCircuit(4,4)
        vqeCirc.x(0)                       
        vqeCirc.x(1)                       
        vqeCirc.x(2)                       
        vqeCirc.x(3)                       
        vqeCirc.rx(theta,2)  # do X(θ)
        vqeCirc.cx(2,1)
        vqeCirc.h(1)
        vqeCirc.rx(np.pi/2,2)  # do X(π/2)
        vqeCirc.cx(1,0)
        vqeCirc.cx(2,3)
        vqeCirc.s(3)
        backend = Aer.get_backend('statevector_simulator')
        result = qiskit.execute(vqeCirc, backend=backend).result()
        psi0 =  result.get_statevector(vqeCirc)
        return (np.conjugate(psi0.T) @ Hamiltonian @ psi0).real
    #Minimizes over the VQE function
    minObj = optimize.minimize_scalar(Psi,tol=1e-8,
                                      method='brent', #bracket=(-np.pi, np.pi)
                                      #method='bounded',bounds=(-np.pi, np.pi),options={'xatol': 1e-08}
                                      )
    param = minObj.x
    #Buidls the circuit that we actually wanted
    def GroundStateCircuit(circuit):
        circuit.x(0)                       
        circuit.x(1)                       
        circuit.x(2)                       
        circuit.x(3)                       
        circuit.rx(param,2)  # do X(θ)
        circuit.cx(2,1)
        circuit.h(1)
        circuit.rx(np.pi/2,2)  # do X(π/2)
        circuit.cx(1,0)
        circuit.cx(2,3)
        circuit.s(3)
        return circuit

    return GroundStateCircuit, minObj


def PauliExpGate(qc,theta, pauliString):
    """
    Given a Quantum Circuit Object, appends the circuit for the Exponential of a Pauli

    Exp
    """
    #print(pauliString)
    #print(theta)
    #pauliString = pauliString[::-1]
    theta = theta.real
    #First Loop iterates over the Pauli String, indexed left to right (0, 1, 2, 4)
    #In addition, stores the indices which are operated on (not zero) so that we can apply CNOTs correctly
    indicesInGate = []
    for i in range(len(pauliString)): #For each qubit in the register:
        if pauliString[i] == 0: #For idenity, do nothing
            pass
        elif pauliString[i] == 1: #For X, rotate the the X basis
            qc.h(i)
            indicesInGate.append(i) 
        elif pauliString[i] == 2: #For Y, rotate the the Y basis
            qc.rx(np.pi/2,i)#u2(np.pi / 2, np.pi / 2,i)
            indicesInGate.append(i)
        else:
            indicesInGate.append(i) #For Z, already in the Z basis
    #Now we implement the CNOT according to the indicies in gate:
    for i in range(len(indicesInGate) - 1): #Only go up the second to last index
        qc.cnot(indicesInGate[i], indicesInGate[i + 1])
    #Perform the Rz rotaation (over 2 theta)
    qc.rz(-2*theta, indicesInGate[-1])
    #Apply the Inverse Ladder
    for i in reversed(range(1,len(indicesInGate))): #The first index is handled by the i-1
        qc.cnot(indicesInGate[i-1], indicesInGate[i])
    #Apply the Basis transformations:
    for i in range(len(pauliString)): #For each qubit in the register:
        if pauliString[i] == 0: #For idenity, do nothing
            pass
        elif pauliString[i] == 1: #For X, rotate the the X basis
            qc.h(i)
        elif pauliString[i] == 2: #For Y, rotate the the Y basis
            qc.rx(-np.pi/2,i)#qc.u2(np.pi / 2, np.pi / 2,i)
        else:
            pass
    #print(qc.draw())
    return qc

def circKHK(circuit, kCoefs, hCoefs, k, h, stabilize=None, barriers=False):
    """
    Creates a Time Evolution Circuit
    """
    if stabilize is None:
        #First, loop over the terms in k (using the adjoint k first)
        #Short K so that CNOTs on Pairs of Terms cancel
        for (term, co) in zip(k[::-1], kCoefs[::-1]):
            if barriers:
                circuit.barrier(range(4))
            circuit = (PauliExpGate(circuit, -1*co, term))
        for (term, co) in zip(h, hCoefs):
            if barriers:
                circuit.barrier(range(4))
            circuit = (PauliExpGate(circuit, -1*co, term))
        for (term, co)  in zip(k, kCoefs):
            if barriers:
                circuit.barrier(range(4))
            circuit = (PauliExpGate(circuit, co, term))
    if stabilize is 'k':
        k, kCoefs = kStabilizer(k, kCoefs)
        #Moves (3,3,0,3) to the second postiion
        index = k.index((3,3,0,3))
        k.insert(0,k[index])
        k.pop(index+1)
        kCoefs.insert(0,kCoefs[index])
        kCoefs.pop(index+1)
        ## Moves (3,0,0,3) to the front
        index = k.index((3,0,0,3))
        k.insert(0,k[index])
        k.pop(index+1)
        kCoefs.insert(0,kCoefs[index])
        kCoefs.pop(index+1)

        circuit.cx(0,1)
        circuit.sdg(0)
        circuit.cx(2,3)
        circuit.sdg(2)
        circuit.h(0)
        circuit.cx(0,2)
        circuit.h(0)
        for (term, co) in zip(k[::-1], kCoefs[::-1]):
            circuit = (PauliExpGate(circuit, -1*co, term))

        circuit.h(0)
        circuit.cx(0,2)
        circuit.h(0)
        circuit.s(2)
        circuit.cx(2,3)
        circuit.s(0)
        circuit.cx(0,1)
        for (term, co) in zip(h, hCoefs):
            circuit = (PauliExpGate(circuit, -1*co, term))

        circuit.cx(0,1)
        circuit.sdg(0)
        circuit.cx(2,3)
        circuit.sdg(2)
        circuit.h(0)
        circuit.cx(0,2)
        circuit.h(0)
        for (term, co)  in zip(k, kCoefs):
            circuit = (PauliExpGate(circuit, co, term))
        circuit.h(0)
        circuit.cx(0,2)
        circuit.h(0)
        circuit.s(2)
        circuit.cx(2,3)
        circuit.s(0)
        circuit.cx(0,1)


        pass
    elif stabilize is 'kh':
        pass
    return circuit

def GBAt(circuit, ParametersObj, time, ctrlgates = ['cx','cx'], evolution='KHK', stabilize=None, barriers=False):
    """
    Creates the circuit for <BA(t)> = G
    Re<X(t)X>: ctrlgates = ['cx','cx']
    Re<Y(t)Y>: ctrlgates = ['cy','cy']
    Re<X(t)Y>: ctrlgates = ['cx','cy']
    Re<Y(t)X>: ctrlgates = ['cy','cx']    
         ┌───┐                     ┌───┐┌─┐
    q_0: ┤ H ├──■──────────────■───┤ H ├┤M├
         └───┘┌─┴──┐┌───────┐┌─┴──┐└───┘└╥┘
    q_1: ─────┤ B  ├┤0      ├┤ A  ├──────╫─
              └────┘│       │└────┘      ║ 
    q_2: ───────────┤1      ├────────────╫─
                    │  U(t) │            ║ 
    q_3: ───────────┤2      ├────────────╫─
                    │       │            ║ 
    q_4: ───────────┤3      ├────────────╫─
                    └───────┘            ║ 
    c_0: ════════════════════════════════╩═    
    """
    sub_qr = QuantumRegister(4)
    sub_qc = QuantumCircuit(sub_qr)
    
    circuit.h(0)
    getattr(circuit,ctrlgates[1])(0,1)
    if evolution == 'KHK':
        sub_qc = circKHK(sub_qc, ParametersObj.kCoefs, np.multiply(ParametersObj.hCoefs, time), ParametersObj.cartan.k, ParametersObj.cartan.h, stabilize=stabilize, barriers=barriers)
        circuit.append(sub_qc, range(1,5))
    else:
        sub_qc = Trotter(sub_qc, ParametersObj.hamiltonian, time, evolution)
        circuit.append(sub_qc, range(1,5))
    getattr(circuit,ctrlgates[0])(0,1)
    circuit.h(0)

    return circuit

def Trotter(circuit, HamiltonianObj, time, steps):
    """
    Prepares a basis Trotterized Circuit from the Hamiltonian Object

    """
    #Appends the Trotter Step circuit for each step
    for i in range(steps):
        timeStep = time/steps
        for (co, term) in zip(HamiltonianObj.HCoefs, HamiltonianObj.HTuples):
            circuit = PauliExpGate(circuit, co * timeStep, term)

    return circuit

def kStabilizer(k, kCoefs):
    #First, we rebuild the elements by building the stabilizer circuit and transforming the existing k elements
    #Second, we transform the coefficients to the new basis by multiplying by the coefficients of the transformed k elements
    #Then, just pass the k back to be compiled normally

    pass
    Kterms = CQS2OFterms(k, kCoefs)
    Klist = np.array([QO(op) for op in Kterms.keys()])
    
    g = OFgate()
    u = []
    u.append(g.h(0)*g.cx(0,2)*g.h(0))
    u.append(g.cx(2,3)*g.s(2))
    u.append(g.cx(0,1)*g.s(0))
    uk = u[2]*u[1]*u[0]
    Qlist = np.array(hermitian_conjugated(uk)) * Klist * np.array(uk)
    knew = []
    kCoefsNew = []
    for (OFterm, kco) in zip(Qlist, kCoefs):
        OFterm.compress()
        coNew, tupNew = OpenFermionToCQS(OFterm, 4)
        knew.append(tupNew[0])
        kCoefsNew.append(np.multiply(coNew[0], kco))
    
    return knew, kCoefsNew
        
def GBAt_manual(circuit, ParametersObj, time, ctrlgates = ['cx','cx']):
    sub_qr = QuantumRegister(4)
    sub_qc = QuantumCircuit(sub_qr)
    kCo = ParametersObj.kCoefs
    k = ParametersObj.cartan.k
    hCo = ParametersObj.hCoefs
    h = ParametersObj.cartan.h

    #Interference on Ancilla
    circuit.h(0)
    getattr(circuit,ctrlgates[1])(0,1)


    #Neighboring K elements
    #Ordered as [(1, 2, 3, 0), SWAP 2-3 (1, 2, 0, 3), (2, 1, 0, 3), SWAP 2-3  (2, 1, 3, 0), (0, 3, 2, 1)  (0, 3, 1, 2) SWAP 0-1 (3, 0, 1, 2) (3, 0, 2, 1) ]
    # This saves 2 swaps on the end and the ordering if the k terms reduces the number of CNOTs
    #The ordering as a list, but reveresed: 
    SwapsForK = [ None, (2,3), None, (2,3), None, None, (0,1), None]
    kDaggerOrdering = [(1, 2, 3, 0), (1, 2, 0, 3), (2, 1, 0, 3), (2, 1, 3, 0), (0, 3, 2, 1),  (0, 3, 1, 2), (3, 0, 1, 2), (3, 0, 2, 1)]
    kDaggerAfterSwaps = [(1, 2, 3, 0), (1, 2, 3, 0), (2, 1, 3, 0), (2, 1, 3, 0), (0, 3, 2, 1),  (0, 3, 1, 2), (0, 3, 1, 2), (0, 3, 2, 1)]

    #Longer Version to see if it works:
    #[(1, 2, 3, 0), SWAP 2-3 (1, 2, 0, 3), (2, 1, 0, 3), SWAP 2-3  (2, 1, 3, 0),  (0, 3, 1, 2) SWAP 0-1 (3, 0, 1, 2) (3, 0, 2, 1) SWAP0-1 (0, 3, 2, 1)]
    SwapsForKDagger = [ None, (2,3), None, (2,3), None, (0,1), None, (0,1)]
    SwapsForK = [ (2,3), None, (2,3), None, (0,1), None, (0,1), None]
    kDaggerOrdering  =  [(1, 2, 3, 0), (1, 2, 0, 3), (2, 1, 0, 3), (2, 1, 3, 0), (0, 3, 1, 2), (3, 0, 1, 2), (3, 0, 2, 1), (0, 3, 2, 1)]
    kDaggerAfterSwaps = [(1, 2, 3, 0), (1, 2, 3, 0), (2, 1, 3, 0), (2, 1, 3, 0), (0, 3, 1, 2), (0, 3, 1, 2), (0, 3, 2, 1), (0, 3, 2, 1)]

    #Sorts k for applying the dagger based on kDaggerOrdering:
    k = [a for a in k]
    kCo = [a for a in kCo]
    for ktup in kDaggerOrdering[::-1]:
        index = k.index(ktup)
        k.insert(0,k[index])
        k.pop(index+1)
        kCo.insert(0,kCo[index])
        kCo.pop(index+1)
    #K is now sorted, and we can apply it to the circuit using the code from , but using the kDaggerAfterSwaps sorting
    #It commutes so we don't care about reversing the the ordering except to preserve the swap ordering
    for (term, swap, co) in zip(kDaggerAfterSwaps,SwapsForKDagger, kCo):
        if swap is not None:
            sub_qc.swap(swap[0],swap[1])
        sub_qc = (PauliExpGate(sub_qc, -1*co, term))
    #We can now apply h. Technically 0 and 1 are swapped, but this doesn't matter to the terms in h:
    for (term, co) in zip(h, hCo):
        sub_qc = (PauliExpGate(sub_qc, -1*co*time, term))
    for (term, swap, co) in zip(kDaggerAfterSwaps[::-1],SwapsForK[::-1], kCo[::-1]):
        if swap is not None:
            sub_qc.swap(swap[0],swap[1])
        sub_qc = (PauliExpGate(sub_qc, co, term))

    circuit.append(sub_qc, range(1,5))
    getattr(circuit,ctrlgates[0])(0,1)
    circuit.h(0)
    return circuit


def GBAt_manual_smallK(circuit, ParametersObj, time, ctrlgates = ['cx','cx'], randomSeed = 0):
    sub_qr = QuantumRegister(4)
    sub_qc = QuantumCircuit(sub_qr)
    kCo = ParametersObj.kCoefs
    k = ParametersObj.cartan.k
    hCo = ParametersObj.hCoefs
    h = ParametersObj.cartan.h

    #Interference on Ancilla
    circuit.h(0)
    

    #Longer Version to see if it works:
    #q0 -> 0, q1 -> 1, q2 -> 2, q3 -> 3
    #k0  [(2, 1, 3, 0), SWAP 2,3 (2, 1, 0, 3),  SWAP 0,1 (3, 0, 2, 1), (3, 0, 1, 2)]
    #k0 after swaps [(2, 1, 3, 0), (2, 1, 3, 0), (0, 3, 1, 2), (0, 3, 2, 1)]
    #q0 -> 1, q1 -> 0, q2 -> 3, q3 -> 2
    #SWAP 0,1
    #CX0-1
    #k1 = (1, 2, 0, 3), SWAP 2-3 (1, 2, 3, 0), SWAP0-1 (0, 3, 1, 2), (0, 3, 2, 1)
    #k1 after swaps (2,1,3,0), (2,1,3,0), (0,3,1,2),(0,3,2,1)
    #q0 -> 0, q1 -> 1, q2 -> 2, q3 -> 3

    SwapsForK0 = [None, (2,3), None, None]
    K0Ordering = [(1, 2, 3, 0), (1, 2, 0, 3), (0, 3, 1, 2), (0, 3, 2, 1)]
    K0Swapped =  [(1,2,3,0), (1,2,3,0), (0,3,2,1),(0,3,1,2)]

    SwapsForK1 = [None, (2,3), (0,1), None]
    k1Ordering = [(2, 1, 0, 3), (2, 1, 3, 0), (3, 0, 2, 1), (3, 0, 1, 2)]
    k1Swapped =  [(2, 1, 3, 0), (2, 1, 3, 0), (0, 3, 2, 1), (0, 3, 1, 2)]

    SwapsForK1Dagger =  [None, None, (0,1), (2,3)]
    k1DaggerOrdering =  [(3, 0, 1, 2), (3, 0, 2, 1), (2, 1, 3, 0), (2, 1, 0, 3)]
    k1DaggerSwapped  =  [(0, 3, 1, 2), (0, 3, 2, 1), (2, 1, 3, 0), (2, 1, 3, 0)]

    #Sorts k for applying the dagger based on kDaggerOrdering:
    k = [a for a in k]
    k0 = [a for a in k if a in K0Ordering]
    k1 = [a for a in k if a in k1Ordering]
    kCo = [a for a in kCo]
    k0Co = [a for a,b in zip(kCo,k) if b in K0Ordering]
    k1Co = [a for a,b in zip(kCo,k) if b in k1Ordering]

    for ktup in K0Ordering[::-1]:
        index = k0.index(ktup)
        k0.insert(0,k0[index])
        k0.pop(index+1)
        k0Co.insert(0,k0Co[index])
        k0Co.pop(index+1)
    for ktup in k1Ordering[::-1]:
        index = k1.index(ktup)
        k1.insert(0,k1[index])
        k1.pop(index+1)
        k1Co.insert(0,k1Co[index])
        k1Co.pop(index+1)

    #K is now sorted, and we can apply it to the circuit using the code from , but using the kDaggerAfterSwaps sorting
    #It commutes so we don't care about reversing the the ordering except to preserve the swap ordering
    for (term, swap, co) in zip(K0Swapped,SwapsForK0, k0Co):
        if swap is not None:
            sub_qc.swap(swap[0],swap[1])
        sub_qc = (PauliExpGate(sub_qc, -1*co, term))

    
    circuit.append(sub_qc, range(1,5))

    getattr(circuit,ctrlgates[1])(0,1)
    sub_qr = QuantumRegister(4)
    sub_qc = QuantumCircuit(sub_qr)
    #Applied interference CNOT
    for (term, swap, co) in zip(k1Swapped,SwapsForK1, k1Co):
        if swap is not None:
            sub_qc.swap(swap[0],swap[1])
        sub_qc = (PauliExpGate(sub_qc, -1*co, term))
    #Randomize the h ordering
    

    for (term, co) in zip(h, hCo):
        sub_qc = (PauliExpGate(sub_qc, -1*co*time, term))
    for ktup in k1DaggerOrdering[::-1]:
        index = k1.index(ktup)
        k1.insert(0,k1[index])
        k1.pop(index+1)
        k1Co.insert(0,k1Co[index])
        k1Co.pop(index+1)  
    
    for (term, swap, co) in zip(k1DaggerSwapped,SwapsForK1Dagger, k1Co):
        if swap is not None:
            sub_qc.swap(swap[0],swap[1])
        sub_qc = (PauliExpGate(sub_qc, co, term))
    circuit.append(sub_qc, range(1,5))
    getattr(circuit,ctrlgates[0])(0,1)
    circuit.h(0)
    return circuit
"""

for (term, co) in zip(h, hCoefs):
    if barriers:
        circuit.barrier(range(4))
    circuit = (PauliExpGate(circuit, -1*co, term))
for (term, co)  in zip(k, kCoefs):
    if barriers:
        circuit.barrier(range(4))
    circuit = (PauliExpGate(circuit, co, term))

"""