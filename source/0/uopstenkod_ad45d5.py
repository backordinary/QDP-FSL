# https://github.com/nynazdravkovic/Quantum-tomography-of-random-entangled-states/blob/9442408599d35e012f8480e3ba8c56026fd623bc/uopstenKod.py
import numpy as np
import itertools
import qiskit 
from qiskit import QuantumRegister, QuantumCircuit,ClassicalRegister, QuantumCircuit,execute
from qiskit import Aer
import qiskit.ignis.verification.tomography as tomo
from qiskit.quantum_info import state_fidelity
import math
from sympy import I, Matrix, symbols
from sympy.physics.quantum import TensorProduct
import matplotlib.pyplot as plt
# unesi broj qubita 
n=5
#kombinacija svih parametra, tip niz nizova intova
def parametri(N):   
    res =[ ele for ele in itertools.product([0,1,2,3], repeat = N)]
    return res
par=parametri(n)

def merX(kpom,i):
    kpom.h(i)
    return kpom

def merY(kpom,i):
    kpom.s(i)
    kpom.h(i)
    return kpom

def plotMapu(matrica, naslov):
    fig, ax = plt.subplots(figsize=(8,8))

    cm=ax.matshow(matrica)
    no_labels = len(matrica) # how many labels to see on axis x
    positions = np.arange(0,no_labels) # pixel count at label position
    labels = stanjastring # labels you want to see
    plt.xticks(positions, labels,rotation=90)
    plt.yticks(positions, labels)

#    for i in range(no_labels):
#       for j in range(no_labels):
#          c = matrica[j,i]

    plt.colorbar(cm)
    plt.title(naslov,y=-0.1)
    plt.savefig(naslov+'.png')
    plt.show()

q=[]
c=[]
wKolo=[]

# ovde pravim sva kola koja cu posle da merim, proverila sam u jupyteru i
# to je to kolo sigurno
for i in range(4**n):
    q.append(QuantumRegister(n))
    c.append(ClassicalRegister(n))
    wKolo.append(QuantumCircuit(q[i],c[i]))
    #ovde stavljamo gejtove
    wKolo[i].x(q[i][1])
    wKolo[i].ry(1.37,q[i][3])
    wKolo[i].cx(q[i][3],q[i][1])
    wKolo[i].ry(0.955,q[i][0])
    wKolo[i].ry(math.pi/4,q[i][4])
    wKolo[i].cx(q[i][1],q[i][0])
    wKolo[i].cx(q[i][3],q[i][4])
    wKolo[i].ry(-0.955,q[i][0])
    wKolo[i].ry(-math.pi/4,q[i][4])
    wKolo[i].cx(q[i][0],q[i][1])
    wKolo[i].cx(q[i][4],q[i][3])
    wKolo[i].ry(math.pi/4,q[i][2])
    wKolo[i].cx(q[i][1],q[i][2])
    wKolo[i].ry(-math.pi/4,q[i][2])
    wKolo[i].cx(q[i][2],q[i][1])
    
    wKolo[i].barrier()

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
    job.append(execute(wKolo[i], backend2, shots=1000))
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

for c in counts:
    for red in stanjastring:
        if (red in c)==False: 
            c.update({red:0})

#Ova funkcija definise da li je u sumi svih verovatnoca + ili -
#prva lista ima predsstavlja indekse parametra, a drugi koja je kombinacija 0 i 1
def znak(i,s):
    rez=1
    for j in range(n):
        if s[j]=='1':
            rez*=-1
    return rez

#ovde se racunaju Stoksovi
S=np.zeros(4**n)
for p in range(4**n):
    if p==0:
        S[0]=1
    else:
        for j in range(2**n):
            S[p]+=znak(par[p],s[j])*counts[p][stanjastring[j]]/1000
            #ovde sam kao uzela da zaokruzim stoksove par, nemamm pojma koliko je to pametno tako da se radi
            #S[p]=round(S[p],2)

#ovde tenzorski mnozimo sigme, ovaj deo koda je specifican za broj qubita!!
j = complex(0,1)
matrica = [[[1,0],[0,1]],[[0,1],[1,0]],[[0,-j],[j,0]],[[1,0],[0,-1]]]
tenzorski=[]
for red in par:
    tenzorski.append(np.kron(matrica[red[0]],np.kron(matrica[red[1]],np.kron(matrica[red[2]],np.kron(matrica[red[3]],matrica[red[4]])))))

densityMatrix=np.zeros((2**n,2**n),dtype=np.complex_)
for i in range (len(S)):
    densityMatrix+=np.dot(S[i],tenzorski[i])
densityMatrix=np.dot(1/(2**n),densityMatrix)
plotMapu(densityMatrix.real,'real part-nasa matrica')
plotMapu(densityMatrix.imag,'imag part-nasa matrica')

#ovo sam uzela sa ibmq
psi=[ 0+0j, 0.447+0j, 0.447+0j, 0+0j, 0.447+0j, 0+0j, 0+0j, 0+0j, 0.447+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0.447+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j ]
psik=np.zeros((2**n,1),dtype=np.complex_)
for i in range(len(psi)):
    psik[i][0]=psi[i]
mat=np.outer(psi,psik)
plotMapu(mat.real,'real part-sa podacima sa ibmqa')
plotMapu(mat.imag,'imag part-sa podacima sa ibmqa')