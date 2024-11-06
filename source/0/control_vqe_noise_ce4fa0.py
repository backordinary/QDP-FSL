# https://github.com/Kesson-C/QITC/blob/c00de8fa86f1212090273534b215b848bacafeb1/Control_VQE_noise.py
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 17:13:37 2021

@author: Kesson Chen
"""
from qiskit.chemistry.drivers import PySCFDriver, UnitsType, HFMethodType, BasisType
from qiskit.chemistry import FermionicOperator
from qiskit.aqua.algorithms import NumPyEigensolver
import matplotlib.pyplot as plt
import numpy as np
from qiskit.providers.aer import AerSimulator,QasmSimulator,StatevectorSimulator
from qiskit import transpile
from qiskit import QuantumCircuit, execute
from qiskit.quantum_info import state_fidelity
from scipy.linalg import expm
from scipy.optimize import minimize
from qiskit.circuit import QuantumCircuit, ParameterVector, Parameter
from qiskit.aqua.operators import WeightedPauliOperator
from qiskit.quantum_info import Pauli
from qiskit.providers.aer.noise import NoiseModel,depolarizing_error
from qiskit.aqua.algorithms import VQE
from qiskit.aqua.components.optimizers import COBYLA
#%%
map_type='jordan_wigner'
noise_model = NoiseModel()
errate=1e-4
error = depolarizing_error(errate, 1)
noise_model.add_all_qubit_quantum_error(error,["u1", "u2", "u3"])
errate=1e-5
error = depolarizing_error(errate, 2)
noise_model.add_all_qubit_quantum_error(error,'cx')
sim_noise = QasmSimulator(noise_model=noise_model)
# sim = QasmSimulator()
I  = np.array([[ 1, 0],
               [ 0, 1]])
Sx = np.array([[ 0, 1],
               [ 1, 0]])
Sy = np.array([[ 0,-1j],
               [1j, 0]])
Sz = np.array([[ 1, 0],
               [ 0,-1]])

def get_qubit_op(dist):
    # driver = PySCFDriver(atom="H .0 .0 .0; H .0 .0 " + str(dist)
    #                       +";H .0 " + str(np.sqrt(3)*dist/2)+" " +str(dist/2)#+";H .0 .0 "+str(dist*3)+";H .0 .0 "+str(dist*4)#+";H .0 .0 "+str(dist*5)
    #                     , unit=UnitsType.ANGSTROM,hf_method=HFMethodType.UHF
    #                     , spin=1,charge=0, basis='sto3g')
    driver = PySCFDriver(atom="H .0 .0 .0; H .0 .0 " + str(dist)#+";H .0 .0 "+str(dist*2)
                          # +";H .0 .0 "+str(dist*3)#+";H .0 .0 "+str(dist*4)#+";H .0 .0 "+str(dist*5)
                         , unit=UnitsType.ANGSTROM,hf_method=HFMethodType.UHF
                        , spin=0,charge=0, basis='sto3g')
    molecule = driver.run()
    repulsion_energy = molecule.nuclear_repulsion_energy
    num_particles = molecule.num_alpha + molecule.num_beta
    num_spin_orbitals = molecule.num_orbitals * 2
    ferOp = FermionicOperator(h1=molecule.one_body_integrals, h2=molecule.two_body_integrals)
    qubitOp = ferOp.mapping(map_type=map_type, threshold=0.00000001)
    #g = groupedFermionicOperator(ferOp, num_particles)
    #qubitOp = g.to_paulis()
    shift =  repulsion_energy
    return qubitOp, num_particles, num_spin_orbitals, shift

def makewave(wavefunction,dele,name):
    n=wavefunction.num_qubits
    param = ParameterVector(name,int(3*n*(n-1)+6*n))
    t=0
    for i in range(n):
        if (t not in dele):
            wavefunction.rx(param[t],i)
        t+=1
    for i in range(n):
        for j in range(n):
            if i!=j and (t not in dele):
                wavefunction.cry(param[t],i,j)
            t+=1
    # for i in range(n):
    #     if (t not in dele):
    #         wavefunction.rx(param[t],i)
    #     t+=1
    return wavefunction
# E3=[]
def L(params,wavefunction):
    a={}
    t=0
    for i in wavefunction.parameters:
        a[i]=params[t]
        t+=1
        
    qc = wavefunction.assign_parameters(a)
    qc.snapshot_statevector('final')
    qc.measure_all()
    circ_noise = transpile(qc, sim_noise)
    noise_result = sim_noise.run(circ_noise, shots=1).result()
    u=noise_result.data(0)['snapshots']['statevector']['final'][0]
    # counts = noise_result.get_counts(qc)
    # u=np.zeros(2**wavefunction.num_qubits)
    # for i in list(counts):
    #     u[int(i,2)]=counts[i]
    # u/=sum(u)
    # u=np.sqrt(u)
    # print((v.conj().dot(Hp.dot(v)).real)+shift)
    E3.append(u.conj().dot(Hp.dot(u)).real+shift)
    return u.conj().dot(Hp.dot(u)).real+shift
    # return -state_fidelity(u,ext)

def dtheta(params,wavefunction,H):
    N=wavefunction.num_parameters
    A=np.zeros([N,N],dtype=np.complex128)
    C=np.zeros(N,dtype=np.complex128)
    phi=Lv(params,wavefunction)
    dpdt=[]
    cp=1/2
    a=np.pi/2
    for i in range(len(params)):
        ptmp1=params.copy()
        ptmp2=params.copy()
        ptmp1[i]+=a
        ptmp2[i]-=a    
        dp=cp*(Lv(ptmp1,wavefunction)-Lv(ptmp2,wavefunction))
        dpdt.append(dp)
    for i in range(len(params)):
        for j in range(len(params)):
            A[i,j]=(dpdt[i].conj().dot(dpdt[j])).real+dpdt[i].conj().dot(phi)*dpdt[j].conj().dot(phi)
    for i in range(len(params)):
        # phi=Lv(params,wavefunction)
        C[i]=(dpdt[i].conj().dot(H.dot(phi))).real
    dx=np.linalg.inv(A.real).dot(-C)
    return dx.real

def Lv(params,wavefunction):
    a={}
    t=0
    for i in wavefunction.parameters:
        a[i]=params[t]
        t+=1
    qc = wavefunction.assign_parameters(a)
    # qc.snapshot_density_matrix('final')
    qc.snapshot_statevector('final')
    qc.measure_all()
    # for i in range(100):
    circ_noise = transpile(qc, sim_noise)
    noise_result = sim_noise.run(circ_noise,shots=1).result()
    u=noise_result.data(0)['snapshots']['statevector']['final'][0]
    # U.append(u)
    # u=sum(U)/10
    # counts = noise_result.get_counts(qc)
    # u=np.zeros(2**wavefunction.num_qubits)
    # for i in list(counts):
    #     u[int(i,2)]=counts[i]
    # u/=sum(u)
    # u=np.sqrt(u)
    return u

# def Lvp(params,wavefunction):
#     a={}
#     t=0
#     for i in wavefunction.parameters:
#         a[i]=params[t]
#         t+=1
#     qc = wavefunction.assign_parameters(a)
#     # qc.snapshot_density_matrix('final')
#     qc.snapshot_statevector('final')
#     qc.measure_all()
#     circ = transpile(qc, sim)
#     result = sim.run(circ, shots=1e3).result()
#     u=result.data(0)['snapshots']['statevector']['final'][0]
#     # counts = noise_result.get_counts(qc)
#     # u=np.zeros(2**wavefunction.num_qubits)
#     # for i in list(counts):
#     #     u[int(i,2)]=counts[i]
#     # u/=sum(u)
#     # u=np.sqrt(u)
#     return u
def commutator(A,B):
    return A.dot(B)-B.dot(A)

def anticommutator(A,B):
    return A.dot(B)+B.dot(A)

def label2Pauli(s): # can be imported from groupedFermionicOperator.py
    """
    Convert a Pauli string into Pauli object. 
    Note that the qubits are labelled in descending order: 'IXYZ' represents I_3 X_2 Y_1 Z_0
    
    Args: 
        s (str) : string representation of a Pauli term
    
    Returns:
        qiskit.quantum_info.Pauli: Pauli object of s
    """
    
    xs = []
    zs = []
    label2XZ = {'I': (0, 0), 'X': (1, 0), 'Y': (1, 1), 'Z': (0, 1)}
    for c in s[::-1]:
        x, z = label2XZ[c]
        xs.append(x)
        zs.append(z)
    return Pauli(z = zs, x = xs)

def str2WeightedPaulis(s):
	s = s.strip()
	IXYZ = ['I', 'X', 'Y', 'Z']
	prev_idx = 0
	coefs = []
	paulis = []
	is_coef = True
	for idx, c in enumerate(s + '+'):
		if idx == 0: continue
		if is_coef and c in IXYZ:
			coef = complex(s[prev_idx : idx].replace('i', 'j'))
			coefs.append(coef)
			is_coef = False
			prev_idx = idx
		if not is_coef and c in ['+', '-']:
			label = s[prev_idx : idx]
			paulis.append(label2Pauli(label))
			is_coef = True
			prev_idx = idx
	return WeightedPauliOperator([[c, p] for (c,p) in zip(coefs, paulis)])

#%%
d=[0.74]*1
vt=[]
cvt=[]
ex=[]
sc=[]
s=[]
dt=1e-1
roop=100
totalt=0
wavefunction = QuantumCircuit(4)
dele=[]
wavefunction.x(0)
wavefunction.x(2)
wavefunction = makewave(wavefunction, dele,1)
N=wavefunction.num_parameters
ds=[]
name=0
for dist in d:
    qubitOp, num_particles, num_spin_orbitals, shift=get_qubit_op(dist)
    ns=2**qubitOp.num_qubits
    Hp=qubitOp.to_opflow().to_matrix()
    Hd=[]
    perms=[]
    coef=[]
    Hd=[]
    Hd.append(str2WeightedPaulis('1IIIZ').to_opflow().to_matrix())
    Hd.append(str2WeightedPaulis('1IIZI').to_opflow().to_matrix())
    Hd.append(str2WeightedPaulis('1IZII').to_opflow().to_matrix())
    Hd.append(str2WeightedPaulis('1ZIII').to_opflow().to_matrix())
    Hd.append(str2WeightedPaulis('1IIZZ').to_opflow().to_matrix())
    Hd.append(str2WeightedPaulis('1IZZI').to_opflow().to_matrix())
    Hd.append(str2WeightedPaulis('1ZZII').to_opflow().to_matrix())
    Hd.append(str2WeightedPaulis('1ZIIZ').to_opflow().to_matrix())
    Hd.append(str2WeightedPaulis('1ZIZI').to_opflow().to_matrix())
    Hd.append(str2WeightedPaulis('1IZIZ').to_opflow().to_matrix())
    ext = NumPyEigensolver(qubitOp).run().eigenstates.to_matrix().reshape(-1)
    u,w=np.linalg.eig(Hp)
    print('Distance: ',dist)
    np.random.seed(1)
    x0=np.random.rand(N)
    x1=x0.copy()
    cv=Lv(x0, wavefunction)
    v=Lv(x1, wavefunction)
    beta=np.zeros(len(Hd),dtype=np.complex128)
    S=0.3
    ex.append(ext.conj().dot(Hp.dot(ext))+shift)
    E1=[v.conj().dot(Hp.dot(v)).real+shift]
    E2=[cv.conj().dot(Hp.dot(cv)).real+shift]
    print('\n fidelity:',state_fidelity(ext, v),state_fidelity(ext, cv))
    on=1
    ###VQE###
    optimizer = COBYLA(maxiter=roop)
    counts = []
    tmp=Lv(x0, wavefunction)
    E3 = [tmp.conj().dot(Hp.dot(tmp)).real+shift]
    def store_intermediate_result(eval_count, parameters, mean, std):
        counts.append(eval_count)
        E3.append(mean+shift)
        print(E3[-1])
    a=VQE(qubitOp,wavefunction,optimizer,callback=store_intermediate_result,initial_point=x0)
    rs=a.run(sim_noise)
    ###########
    for i in range(roop):
        H=Hp.copy()
        if on:
            for k in range(len(Hd)):
                A=(cv.conj().dot(anticommutator(Hd[k],Hp)).dot(cv)
                      -2*cv.conj().dot(Hp.dot(cv))*cv.conj().dot(Hd[k].dot(cv)))
                beta[k]=(2*S/(1+np.exp(-5*A.real))-S)
                H+=beta[k]*Hd[k]
            print(np.linalg.norm(H),np.linalg.norm(Hp))
        dx0=dtheta(x0, wavefunction, H*dt)
        x0+=dx0
        cv=Lv(x0, wavefunction)
        dx1=dtheta(x1, wavefunction, Hp*dt)
        x1+=dx1
        v=Lv(x1, wavefunction)
        E1.append(v.conj().dot(Hp.dot(v)).real+shift)
        E2.append(cv.conj().dot(Hp.dot(cv)).real+shift)
        print(i,'\n Energy:',E1[-1],E2[-1])

    # 
    # np.savetxt('paperdata/text/N'+str(name)+'E1.txt',E1)
    # np.savetxt('paperdata/text/N'+str(name)+'E2.txt',E2)  
    # np.savetxt('paperdata/text/N'+str(name)+'E3.txt',E3)   
    name+=1
    # np.savetxt('paperdata/text/NFE1.txt',E1)
    # np.savetxt('paperdata/text/NFE2.txt',E2)
    # np.savetxt('paperdata/text/NFE3.txt',E3)
#%%plot
# from qiskit.tools.visualization import circuit_drawer
# circuit_drawer(wavefunction, output='mpl', style={'backgroundcolor': '#FFFFFF'})
# from scipy.stats import t as tCI
# ex=np.loadtxt('paperdata/text/ex.txt',dtype='complex')[0].real
# DE1=[]
# DE2=[]
# DE3=[]
# dt=1e-1
# num=20
# for i in range(num):
#     DE1.append(np.loadtxt('paperdata/text/N'+str(i)+'E1.txt'))
#     DE2.append(np.loadtxt('paperdata/text/N'+str(i)+'E2.txt'))
#     DE3.append(np.loadtxt('paperdata/text/N'+str(i)+'E3.txt'))
# NFE1=np.loadtxt('paperdata/text/NFE1.txt')
# NFE2=np.loadtxt('paperdata/text/NFE2.txt')
# NFE3=np.loadtxt('paperdata/text/NFE3.txt')
# t=np.array(list(range(len(NFE3))))*dt
# dof = num-1 
# confidence = 0.997
# t_crit = np.abs(tCI.ppf((1-confidence)/2,dof))
# meanE1=np.zeros(len(DE1[0]))
# for i in DE1:
#     meanE1+=i
    
# meanE1/=num

# meanE2=np.zeros(len(DE2[0]))
# for i in DE2:
#     meanE2+=i
    
# meanE2/=num

# meanE3=np.zeros(len(DE3[0]))
# for i in DE3:
#     meanE3+=i
    
# meanE3/=num

# varE1=[]
# varE2=[]
# varE3=[]
# for i in range(len(DE1[0])):
#     tmax=[]
#     for j in DE1:
#         tmax.append((j[i]).real)
#     varE1.append((abs(min(tmax)-meanE1[i]),abs(max(tmax)-meanE1[i])))
#     tmax=[]
#     for j in DE2:
#         tmax.append((j[i]).real)
#     varE2.append((abs(min(tmax)-meanE2[i]),abs(max(tmax)-meanE2[i])))
#     tmax=[]
#     for j in DE3:
#         tmax.append((j[i]).real)
#     varE3.append((abs(min(tmax)-meanE3[i]),abs(max(tmax)-meanE3[i])))

# NFE1=NFE1-ex
# NFE2=NFE2-ex
# NFE3=NFE3-ex
# meanE1=meanE1-ex
# meanE2=meanE2-ex
# meanE3=meanE3-ex

# plt.errorbar(t,meanE3,np.array(varE3).T,fmt='g^',alpha=0.7, capsize=10)
# plt.plot(t,NFE1,'b-',label='Noise Free Imaginary Time',linewidth=2)
# plt.plot(t,NFE2,'r-',label='Noise Free Imaginary Time Control',linewidth=2)
# plt.plot(t,NFE3,'g-',label='Noise Free VQE (COBYLA)',linewidth=2)
# plt.plot(t,NFE1*0+1e-3,'k',label='Chemical Accuracy',linewidth=3)
# plt.plot(t,meanE1,'b--',label='Mean Noisy Imaginary Time',linewidth=2)
# plt.errorbar(t,meanE1,np.array(varE1).T,fmt='b^',alpha=0.7, capsize=10)
# plt.plot(t,meanE2,'r--',label='Mean Noisy Imaginary Time Control',linewidth=2)
# plt.errorbar(t,meanE2,np.array(varE2).T,fmt='r^',alpha=0.7, capsize=10)
# plt.plot(t,meanE3,'g--',label='Mean Noisy VQE (COBYLA)',linewidth=2)
# plt.xticks(size=20)
# plt.yticks(size=20)
# plt.xlim([-0.05,5.05])
# plt.ylim([1e-4,1])
# plt.ylabel('Energy(Hartree)', size=25,labelpad=20)
# plt.xlabel('τ', size=25)
# plt.yscale('log')
# plt.title('Variational Ansatz-Based',size=25)
# plt.legend(fontsize='xx-large',loc='lower left',
#             fancybox=True, shadow=True, ncol=2)