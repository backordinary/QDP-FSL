# https://github.com/nynazdravkovic/Quantum-tomography-of-random-entangled-states/blob/9442408599d35e012f8480e3ba8c56026fd623bc/randomKola/nasa-tomografija-pokusaj-random-stanja-1.py
import numpy as np
import itertools
import qiskit 
from qiskit import QuantumRegister, QuantumCircuit,ClassicalRegister, QuantumCircuit,execute
from qiskit import Aer
import qiskit.ignis.verification.tomography as tomo
import math
import matplotlib.pyplot as plt
import random
import matplotlib as mpl
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import pauli_error
import qiskit.quantum_info
from qiskit.ignis.mitigation.measurement import (complete_meas_cal,CompleteMeasFitter)

#fig, ax = plt.subplots(figsize=(0.5,8))
#fig.subplots_adjust()
cmap = mpl.cm.cool
norm = mpl.colors.Normalize(vmin=-0.1, vmax=0.5)

#fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
             #cax=ax, orientation='vertical')
from scipy.linalg import sqrtm
def fidelity(rho1,rho2):
    a=np.matmul(rho2,sqrtm(rho1))
    b=np.matmul(sqrtm(rho1),a)
    c=sqrtm(b)
    tr = 0
    tr+=c[0][0]
    tr+=c[1][1]
    tr+=c[2][2]
    tr+=c[3][3]
    tr+=c[4][4]
    tr+=c[5][5]
    tr+=c[6][6]
    tr+=c[7][7]
    return(tr**2)
def get_noise(p):
    error_meas = pauli_error([('X',p), ('I', 1 - p)])
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_meas, "measure") # measurement error is applied to measurements        
    return noise_model


n=3
#broj single qubit kapija
n_s=10
#broj multi qubit kapija
n_m=4
#sum
noise_model = get_noise(0.15)


#kombinacija svih parametra, tip niz nizova intova
def parametri(N):   
    res =[ ele for ele in itertools.product([0,1,2,3], repeat = N)]
    return res
par=parametri(n)

#fje za merenje u osama x i y
def merX(kpom,i):
    kpom.h(i)
    return kpom
def merY(kpom,i):
    kpom.s(i)
    kpom.h(i)
    return kpom

#fja koj dodaje random kapije na kolo 
def randomCirc(circ_list, n, n_single_qubit_gate, n_two_qubit_gate):
    niz=n_single_qubit_gate*[1] + n_two_qubit_gate*[2]
    random.shuffle(niz)
    for i in range(n_single_qubit_gate+n_two_qubit_gate):
        if niz[i]==1:
            qubit=random.randint(0, n-1)
            gate=random.randint(0,2)
            if gate==0:
                for circ in circ_list:
                    circ.h(qubit)
            if gate==1:
                for circ in circ_list:
                    circ.s(qubit)
            if gate==2:
                for circ in circ_list:
                    circ.t(qubit)
        if niz[i]==2:
            two_qubits=random.sample(range(0, n), 2)
            for circ in circ_list:
                circ.cx(two_qubits[0], two_qubits[1])
#Ova funkcija definise da li je u sumi svih verovatnoca + ili -
#prva lista ima predsstavlja indekse parametra, a drugi koja je kombinacija 0 i 1
def znak(i,s):
    rez=1
    for j in range(n):
        if s[j]=='1':
            rez*=-1
    return rez

def plotMapu(matrica, naslov):
    fig, ax = plt.subplots(figsize=(8,8))
    cm=ax.matshow(matrica, cmap=plt.cm.Blues)
    no_labels = len(matrica) # how many labels to see on axis x
    positions = np.arange(0,no_labels) # pixel count at label position
    labels = stanjastring # labels you want to see
    plt.xticks(positions, labels,rotation=90)
    plt.yticks(positions, labels)
    fig.colorbar(cm)#,mpl.cm.ScalarMappable(norm=norm, cmap=cm), orientation='vertical')
    plt.title(naslov,y=-0.1)

nizFid=[]
for abc in range(100):
    print(abc)
    #ovde unosimo gejtove
    q=[]
    c=[]
    wKolo=[]
    
    # sad jednostavno zelim da budu svi isti gejtovi
    for i in range(4**n):
        q.append(QuantumRegister(n))
        c.append(ClassicalRegister(n))
        wKolo.append(QuantumCircuit(q[i],c[i]))
    randomCirc(wKolo,n,n_s,n_m)
    for i in range(4**n):
        wKolo[i].barrier()
    
    
    
    result = execute(wKolo[0], Aer.get_backend('statevector_simulator')).result()
    psi  = result.get_statevector(wKolo[0])
    psik=np.zeros((2**n,1),dtype=np.complex_)
    for i in range(len(psi)):
        psik[i][0]=np.conj(psi[i])
    mat=np.outer(psi,psik)
    pokusaj=qiskit.quantum_info.DensityMatrix(psi).data
    
    #dodajemo merenja na svaki nacin
    for i in range(4**n):
        for j in range(n):
            if par[i][j]==1:
                merX(wKolo[i],j)
                wKolo[i].measure(q[i][j],c[i][j])
            if par[i][j]==2:
                merY(wKolo[i],j)
                wKolo[i].measure(q[i][j],c[i][j])
            if par[i][j]==3:
                wKolo[i].measure(q[i][j],c[i][j])
                     
    
    #izvrsavamo kola
    backend2 = Aer.get_backend('qasm_simulator')
    job=[]
    result=[]
    counts=[]
    for i in range (4**n):
        job.append(execute(wKolo[i], backend2, shots=1000,noise_model=noise_model))
        result.append(job[i].result())
        counts.append(result[i].get_counts(wKolo[i]))
    
    
    s= [ele for ele in itertools.product(['0','1'], repeat = n)]
    stanjastring=[]
    for i in s:
        pom=''
        pom=i[0]
        for j in range(1,len(i)):
            pom+=i[j]
        stanjastring.append(pom)
    #ovde se updejtuju countsi da tamo gde nije izmerio nista ne bude nan nego 0
    for c in counts:
        for red in stanjastring:
            if (red in c)==False: 
                c.update({red:0})
    
    #ovde se racunaju Stoksovi
    S=np.zeros(4**n)
    for p in range(4**n):
        if p==0:
            S[0]=1
        else:
            for j in range(2**n):
                S[p]+=znak(par[p],s[j])*counts[p][stanjastring[j]]/1000
                #S[p]=round(S[p],2)#ovde sam kao uzela da zaokruzim stoksove par, nemamm pojma koliko je to pametno tako da se radi
    
    # ovde za tenzorski se isto menja
    j = complex(0,1)
    matrica = [[[1,0],[0,1]],[[0,1],[1,0]],[[0,-j],[j,0]],[[1,0],[0,-1]]]
    tenzorski=[]
    for red in par:
        tenzorski.append(np.kron(matrica[red[2]],np.kron(matrica[red[1]],matrica[red[0]])))
    densityMatrix=np.zeros((2**n,2**n),dtype=np.complex_)
    for i in range (len(S)):
        densityMatrix+=np.dot(S[i],tenzorski[i])
    
    densityMatrix=np.dot(1/(2**n),densityMatrix)
    densityMatrix1=np.conj(densityMatrix)
    # plotMapu(densityMatrix.real,'real part-nasa matrica')
    plotMapu(mat.real,'real part-sa podacima sa ibmqa')
    #plotMapu(mat.imag,'imag part-sa podacima sa ibmqa')
   #plotMapu(pokusaj.real,'real part-sa podacima sa ibmqa')
    print(fidelity(densityMatrix1,mat))
    
    nizFid.append(fidelity(mat,densityMatrix1))
    
np.save('fidelityZa0.15ns=10nm=4', nizFid)
print('Srednja vrednost je')
print(np.mean(nizFid))
print('Standardna devijacija je')
print(np.std(nizFid))


