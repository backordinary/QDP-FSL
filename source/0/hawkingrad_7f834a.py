# https://github.com/Tihulu/400-Project-Cagil-Benibol/blob/add2ed1b980913eff8ef74ce39177acca4312e91/hawkingrad.py
import numpy as np
import math as m
import cmath as cm
from numpy.linalg import eig
import pennylane as qml
import matplotlib.pyplot as plt

#initials
n=4
Qz = np.zeros((n,n))
Pz = np.zeros((n,n))
a = np.zeros((n,n))
adag = np.zeros((n,n))
for i,j in zip(range(n),range(1,n)):
        a[i][j] = m.sqrt(j)
        adag[j][i] = m.sqrt(j)
        
        
        
# Position basis
#Qpos Ppos
Qpos = np.zeros((n,n),np.complex_)
Ppos = np.zeros((n,n),np.complex_)
F = np.zeros((n,n),dtype=np.complex_)
Ft = np.zeros((n,n),dtype=np.complex_)
Ppos = np.zeros((n,n),dtype=np.complex_)

#Qpos Ppos
for l in range(n):
    Qpos[l][l] = np.sqrt(np.pi/(2*n)) * ((-n/2) + l)
    for k in range(n):
        F[l][k]=np.exp( (((2*np.pi*complex(0,1))*l*k)/n))/(np.sqrt(n))  

Ppos = np.matmul(Qpos,F) 
Ppos = np.matmul(F.conj().T,Ppos)       
aaaas = Ppos / ( np.sqrt(np.pi/(64*2)) )


#oscilatory basis
I=np.identity(n)
Posc = (1/(m.sqrt(2))) * (-(complex(0,1)))  * (a - adag)
Qosc = (1/(m.sqrt(2))) * (a + adag)

Q2=np.matmul(Qosc,Qosc)
Pcomplex=np.conjugate(Posc)
P2=np.matmul(Posc,Posc)

#Position basis 3D
Ip=np.identity(2)

x_ = np.kron( np.kron(Qpos,Ip) , Ip)
y_ = np.kron( np.kron(Ip,Qpos) , Ip)
z_ = np.kron( np.kron(Ip,Ip) , Qpos)

Ppos2 = np.matmul(Ppos,Ppos)

Px2_ = np.kron( np.kron(Ppos2,Ip) , Ip) 
Py2_ = np.kron( np.kron(Ip,Ppos2) , Ip)
Pz2_ = np.kron( np.kron(Ip,Ip) , Ppos2) 


#qiskit vqe calculation import libraries
from qiskit import Aer
from qiskit.opflow import X, Z, I, Y
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import SLSQP, SPSA
from qiskit.circuit.library import TwoLocal
from qiskit.opflow import PauliExpectation

# Hamiltonian
#Which is Given in Planck units
G=1
mass=5
radius=1
HmtrxP = (Px2_/2 + Py2_/2 + Pz2_/2) 

Hp = ((0.19634954084936204) * X ^ I ^ I ^ I) + \
((0.3926990816987241)  * I ^ I ^ I ^ X ) + \
((0.3926990816987241)  * I ^ I ^ X ^ X) + \
((0.39269908169872414) * I ^ X ^ X ^ I) + \
((0.39269908169872414) * X ^ X ^ I ^ I) + \
((0.5890486225480861)  * I ^ I ^ X ^ I) + \
((0.5890486225480861)  * I ^ X ^ I ^ I) + \
((1.7671458676442584)  * I ^ I ^ I ^ I)


result_aer=[]
result_st=[]
result_t=[]
'''
for M in np.arange(1,15,1):
    Hmtrxcr = ( (0.5 * (1 + ((G*M)/(2*radius)) )**(1/4) ) * Hp )
'''

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit.library import EfficientSU2


'''
def get_var_form(params):
    qr = QuantumRegister(1, name="q")
    cr = ClassicalRegister(1, name='c')
    qc = QuantumCircuit(qr, cr)
    qc.u3(params[0], params[1], params[2], qr[0])
    qc.ryparams[0], params[1], params[2], qr[0]()
    qc.measure(qr, cr[0])
    return qc
'''

for r in np.arange(0.1,4,0.1):
    #print(type(r))
    r = float(r)
    Hmtrxcm = (0.5 * (1 + ((G*mass)/(2*r)) )**(1/4) ) * Hp 
    
    
    Hp = Hmtrxcm
    #SLSQP
    
    seed = 50
    #ansatz = TwoLocal(rotation_blocks='ry', entanglement_blocks='cz')
    #ansatz =  EfficientSU2(4, reps=1)
    ansatz = TwoLocal(4, ['rx','ry','rz'], 'cz', 'full', reps=1, insert_barriers=True)
    
    slsqp = SLSQP(maxiter=1000)
    
    qi = QuantumInstance(Aer.get_backend('aer_simulator'), seed_transpiler=seed, seed_simulator=seed)
    
    vqe = VQE(ansatz, optimizer=slsqp, quantum_instance=qi)
    result = vqe.compute_minimum_eigenvalue(operator=Hp)
    #print(result.eigenvalue)
    result_aer.append(result.eigenvalue)
    
    optimizer_evals = result.optimizer_evals
    
    initial_pt = result.optimal_point
    
    algorithm_globals.random_seed = seed
    qi = QuantumInstance(Aer.get_backend('statevector_simulator'), seed_transpiler=seed, seed_simulator=seed)
    
    #ansatz = TwoLocal(rotation_blocks='ry', entanglement_blocks='cz')
    #ansatz = TwoLocal(rotation_blocks='rx'+'ry'+'rz', entanglement_blocks='cx')
    #ansatz =  EfficientSU2(4, reps=1)
    ansatz = TwoLocal(4, ['rx','ry','rz'], 'cx', 'full', reps=1, insert_barriers=True)

    
    
    slsqp = SLSQP(maxiter=1000)
    vqe = VQE(ansatz, optimizer=slsqp, initial_point=initial_pt, quantum_instance=qi)
    result1 = vqe.compute_minimum_eigenvalue(operator=Hp)
    #print(result1)
    optimizer_evals1 = result1.optimizer_evals
    #print()
    #print(f'optimizer_evals is {optimizer_evals1} with initial point versus {optimizer_evals} without it.')
    
    #print(result1.eigenvalue)
    result_st.append(result1.eigenvalue)


for r in np.arange(0.1,4,0.1):

    r = float(r)
    Hmtrxcm = (0.5 * (1 + ((G*mass)/(2*r)) )**(1/4) ) * (Px2_/2 + Py2_/2 + Pz2_/2) 
    #Theoretical calculation for eigenvalues
    e_val1,e_vec = eig(np.nan_to_num(Hmtrxcm))
    result_t.append(min(e_val1))
    



'''
 (0.09817477042468102) [X0 I1 I2 I3]
+ (0.19634954084936204) [I0 I1 I2 X3]
+ (0.19634954084936204) [I0 I1 X2 X3]
+ (0.19634954084936207) [I0 X1 X2 I3]
+ (0.19634954084936207) [X0 X1 I2 I3]
+ (0.29452431127404305) [I0 I1 X2 I3]
+ (0.29452431127404305) [I0 X1 I2 I3]
+ (0.8835729338221292) [I0 I1 I2 I3]
'''

#plot
x = np.arange(0.1,4,0.1)
y = result_st
yt = result_t
plt.plot(x,y,'ro',label='VQE Result')
plt.plot(x,y,'-r')
plt.plot(x,yt,'bo',label='theoretical Result')
plt.plot(x,yt,'-b')

plt.legend(loc="upper left")
plt.title('radial distance vs energy statevector')
plt.xlabel('radial distance')
plt.ylabel('Energy')
plt.show()