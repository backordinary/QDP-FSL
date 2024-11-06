# https://github.com/aubreycoffey/TSP-Quantum-Computing/blob/c78786a4ba3e05cd693d91c7dd78df2676f48906/quantum_algorithms/gsaoa.py
import numpy as np
from qiskit import QuantumCircuit
from qiskit import Aer, execute
from qiskit.circuit import Parameter
from qiskit.circuit.library import CZGate
from qiskit.opflow import I, X, Z, CZ
from qiskit.opflow import PauliTrotterEvolution, Suzuki
from qiskit.quantum_info.operators import Operator, Pauli
from qiskit.circuit import Parameter
from numpy import sqrt
from qiskit.circuit import ParameterVector
import qiskit.quantum_info as qi
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.circuit import QuantumCircuit
from scipy.optimize import minimize
import random

def gsaoa(adj,p,param_seed,initstate):
    def powerset(s):
        se=[]
        x = len(s)
        for i in range(1 << x):
            se.append([s[j] for j in range(x) if (i & (1 << j))])
        return se
    def mixer_hamiltonian(nqubits,beta):
        gate = CZGate()
        qc_mix = QuantumCircuit(nqubits)
        H = QuantumCircuit(nqubits)
        for i in range(0, nqubits):
            for j in range(i+1,nqubits):
                H.append(gate,[i, j])
        op = qi.Operator(H)
        hi=op.data
        qc_mix.hamiltonian(hi, beta, [i for i in range(0, nqubits)])
        return qc_mix
    def problem_hamiltonian(adj,gamma):
        nqubits=len(adj)
        #set penalty, needs to be tested
        pen=int(adj.sum()*10)
        #h1 does cost objective
        H1=0
        st=np.array([I])
        for m in range(1,nqubits):
            st=np.append(st, [I])

        for i in range(0,nqubits):
            for j in range(0,nqubits):
                for k in range(0,nqubits):
                    if j!=i and i!=k and k!=j:           
                        c=.5*(adj[i,k]+adj[i,j])
                        nt=np.copy(st)
                        nt[i]=X
                        nt[j]=Z
                        nt[k]=Z
                        nh=nt[0]
                        for p in range(1,len(nt)):
                            nh=nh^nt[p]
                        nh=c*nh
                        H1=H1+nh             

        #h2 does enforcing of 2 edges
        H2=0
        for i in range(0,nqubits):
            nt=np.copy(st)
            nt[i]=X
            #do no edge penalty
            nh=nt[0]
            for p in range(1,len(nt)):
                nh=nh^nt[p]
            H2=H2+nh

            #do 3...n-1 edge penalty
            vert=list(range(0,nqubits))
            vert.remove(i)
            pos=powerset(vert)
            pows=[i for i in pos if len(i)!=2 and len(i)!=0]
            for m in pows:
                ant=np.copy(nt)
                for j in m:
                    ant[j]=Z
                nh=ant[0]
                for p in range(1,len(ant)):
                    nh=nh^ant[p]
                H2=H2+nh
        H2=pen*H2

        #h3 does subtour breaking
        H3=0
        vert=list(range(0,nqubits))
        pows=powerset(vert)
        pos=[i for i in pows if len(i)<nqubits-2 and len(i)>2]
        for j in pos:
            nt=np.copy(st)
            for a in j:
                nt[a]=X
            nh=nt[0]
            for p in range(1,len(nt)):
                nh=nh^nt[p]
            H3=H3+nh
        H3=pen*H3

        H=H1+H2+H3

        evolution_op = (gamma * H).exp_i() 
        trotterized_op = PauliTrotterEvolution(trotter_mode = Suzuki(order = 1)).convert(evolution_op)
        qc_p = trotterized_op.to_circuit()
        return qc_p

    def measure(qc,nqubits,shots=1024):
        backend = Aer.get_backend('statevector_simulator')
        ss=qc.snapshot('sv', snapshot_type='statevector', qubits=range(0, 4), params=None)
        result = execute(qc, backend,shots=1024,seed_transpiler=123,seed_simulator=123).result()
        snapshots = result.data()['snapshots']['statevector']
        vec=snapshots['sv'][0]
        ed=np.zeros(len(vec))
        for i in range(0,len(vec)):
            if vec[i]<0:
                ed[i]=1  
        edges=[]
        basis=[]
        for m in range(0,nqubits):
            basis.append(2**m)
        basis.reverse()
        for i in range(0,nqubits):
            for j in range(i+1,nqubits):
                ind=basis[i]+basis[j] 
                if ed[ind]==1:
                    edges.append((i,j))             
        return edges

    def compute_expectation(circ,nqubits,adj):
        def find_next(a,el):
            for i in el:
                if a in i:
                    x=i[0]
                    y=i[1]
                    if x!=a:
                        nex=x
                    if y!=a:
                        nex=y
                    el.remove(i)
            return nex,el
        
        edges=measure(circ,nqubits)
        deg2=[]
        qubs=range(0,nqubits)
        for i in qubs:
            count=0
            for j in edges:
                if i in j:
                    count=count+1
            if count==2:
                deg2.append(i)     
        obj = 0
        for i in deg2:
            nbr=[]
            for j in edges:
                if i in j:
                    nbr.append(j)      
            nbrs=[]
            for a in nbr:
                b=a[0]
                c=a[1]
                if b!=i:
                    nbrs.append(b)
                if c!=i:
                    nbrs.append(c)      
            obj=obj+(.5*(adj[i,nbrs[0]]+adj[i,nbrs[1]]))
        #adds penalty for non tours     
        pen=int(adj.sum()*10)
        #non degree two vertices, will add +1 whenever a vertex doesnt have degree two (times the penalty as well) 
        nondeg2=[i for i in qubs if i not in deg2]
        obj=obj+(len(nondeg2)*pen)
        #subtour breaking, will add +1 for each stabilizer operator corresponding to a subtour
        #will add +1 for each subtour
        if len(deg2)==len(qubs) and len(qubs)>5:
            el=edges.copy()
            tours=0
            while len(el)>0:
                a=el[0]
                b=a[0]
                c=a[1]
                el.remove(a)
                while b!=c:
                    b,el=find_next(b,el)
                tours=tours+1
            obj=obj+((len(tours)-1)*pen) 

        return obj

    # We will also bring the different circuit components that
    # build the aoa circuit under a single function
    def create_qaoa_circ(adj, theta,qc_0):
        nqubits = len(adj)
    
        p = len(theta)//2  # number of alternating unitaries
        betas = theta[:p]
        gammas = theta[p:]

        gamma = ParameterVector('G', p)
        beta = ParameterVector('B', p)
        qc = QuantumCircuit(nqubits)
        for irep in range(0, p):       
            # problem unitary
            qc_p=problem_hamiltonian(adj, gamma[irep])
            qc.append(qc_p, [i for i in range(0, nqubits)])

            # mixer unitary
            qc_mix=mixer_hamiltonian(nqubits,beta[irep])
            qc.append(qc_mix, [i for i in range(0, nqubits)])

            qc.assign_parameters({gamma[irep]: gammas[irep]}, inplace = True)
            qc.assign_parameters({beta[irep]: betas[irep]}, inplace = True)
        return qc


    def get_expectation(adj, p,qc_0):  
        def execute_circ(theta):        
            qc = create_qaoa_circ(adj, theta,qc_0)
            return compute_expectation(qc,len(adj),adj)   
        return execute_circ

    nqubits=len(adj)
    random.seed(param_seed)
    theta=[]
    for i in range(2*p):
        x=random.uniform(0, 1)
        if i%2==0:
            x=x*np.pi
        else:
            x=x*2*np.pi
        theta.append(x)
    #run qaoa
    qc_0 = QuantumCircuit(nqubits)
    for i in range(0, nqubits):
        qc_0.h(i)

    if initstate=='shape1':
        qc_0.cz(0,2)
        qc_0.cz(2,1)
        qc_0.cz(1,3)
        qc_0.cz(3,0)        
    if initstate=='shape2':
        qc_0.cz(0,1)
        qc_0.cz(1,3)
        qc_0.cz(3,2)
        qc_0.cz(2,0)
    if initstate=='shape3':
        qc_0.cz(0,1)
        qc_0.cz(1,2)
        qc_0.cz(2,3)
        qc_0.cz(3,0)
    expectation = get_expectation(adj, p,qc_0)
    res = minimize(expectation, theta, method='COBYLA')

    qcfin = create_qaoa_circ(adj, res.x,qc_0)
    finstate=measure(qcfin,len(adj),shots=1024)
    expec=compute_expectation(qcfin,4,adj)

   
    return finstate,expec,res.x







