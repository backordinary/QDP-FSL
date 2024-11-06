# https://github.com/XXChain/Research-Project-/blob/c25bbef38b19580a87b289fa259bf39c41d82a86/functions.py
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 16:55:33 2021

@author: User
"""
import numpy as np
import json
from numpy import identity, kron, shape,matmul
from numpy.linalg import matrix_power
from scipy.sparse.linalg import expm
import matplotlib.pyplot as plt
from constants import N,Jx,Jy,Jz,h_list,S,XX,YY,ZZ,Z,f_s,zeros,q_i,K,Binary,Psi_0,J,g,account,M
#from mitiq import Executor, Observable, PauliString, QPROGRAM, QuantumResult
#from mitiq.interface import mitiq_qiskit
from qiskit.ignis.mitigation.measurement import complete_meas_cal, CompleteMeasFitter
import qiskit.result.result as Result 
from qiskit.providers.ibmq import least_busy
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import pauli_error, depolarizing_error
from qiskit.visualization import plot_histogram, array_to_latex
from qiskit import QuantumCircuit,QuantumRegister, transpile, Aer, IBMQ,assemble,execute
from qiskit.providers.aer import QasmSimulator
from qiskit.ignis.mitigation.measurement import complete_meas_cal, CompleteMeasFitter
from qiskit.providers.ibmq import least_busy
from qiskit.providers.aer.noise.errors import pauli_error, depolarizing_error

def load_account(account):
    IBMQ.save_account(account)
    IBMQ.load_account()
    provider = IBMQ.get_provider('ibm-q')
    return provider

def small_devices(provider):
    load_account(account)
    small_devices = provider.backends(filters=lambda x: x.configuration().n_qubits == 5
                                and not x.configuration().simulator)

    backendQ=least_busy(small_devices)
    return backendQ

def get_backend(name):
    load_account(account)
    return provider.get_backend(name)

def noise_model(backendQ):
    noise_modelQ=NoiseModel.from_backend(backendQ)
    return noise_modelQ

def inital(bits): #intital state 
    f_s=Binary(bits)
    Psi_0=np.zeros(K,)
    Psi_0[f_s]=1
    return Psi_0
def sig(n,N,sig): #finding matirx without using for loop
    A=np.identity(int(2**(n)))
    B=np.identity(int(2**(N-n)))
    return kron(A,kron(sig,B))
def Exp(u,v): #finds expectation values <v|u|v>
    assert(len(shape(u))==2) #check is square 
    #assert(len(v)==len(u[0,:])) #check same no of values
    return np.dot(-np.conj(v),np.matmul(u,v)) 
def fidelity(Phi_t,Phi_Q,m):
    fidelity=[]
    for i in range(m):
        p=Phi_t[:,i] 
        q=Phi_Q[:,i]
        fidelity.append(round(np.square(abs(np.dot(p,q))),10))#|<P|Q>|**2
    return fidelity 
def Exp_z(Phi_t): #finding sig_Z values at each value of |phi(t)>
    m=int(len(Phi_t[0,:]))
    Exp_z_t=zeros(m,N)
    for i in range(N):  
        q=np.empty((m,),dtype=np.complex128) 
        for j in range(m):
            q[j]=Exp(sig(i,N-1,Z),Phi_t[:,j]) 
        Exp_z_t[:,i]=q 
    return Exp_z_t
def Odd_Even(N):
    odd=[]
    even=[]
    for i in range(N):
        if i%2==0:
            even.append(i)
        else:
            odd.append(i)
    return odd, even   
def Index(N,sign): #creates jth index list for producing products 
    index=[]
    odd,even =Odd_Even(N)
    if sign == 1:
        index=odd
    elif sign ==2: 
        index=even
    else:
        index=q_i
    return index 
def form_matrix(spin,sign,J,label): #forms dictionary of decomposed matrices
    assert(len(spin[0,:]))==len(spin[:,0])
    index = Index(N,sign)
    Dict = {}
    if np.shape(spin)[0]==4:
        for i in index:
            for k,v in zip([label+str(i)],[J*sig(i,N-2,spin)]):
                Dict[k]=v
    elif np.shape(spin)[0]==2 and  sign==-1: #if we are using random external magnetic feild 
        for i in index:
          for k,v in zip([label+str(i)],[h_list[i]*sig(i,N-1,spin)]):
              Dict[k]=v
    elif np.shape(spin)[0]==2:
        for i in range(N):
          for k,v in zip([label+str(i)],[J*sig(i,N-1,spin)]):
              Dict[k]=v
    return Dict
def Prod(A,label,t,m,c,steps): #products e^-J*Gate(i)*(delta_t*c) c being 1 or 1/2 
    A_prod=identity(int(2**N)) 
    for i in steps: #steps being odd list even list or list of range N 
        A_prod=np.matmul(expm(-1j*A[label + str(i)]*(t/m)*c),A_prod) 
    return A_prod

def Steps(H,m): #creates dictionary of matrices for each time step
    U_t ={}
    for j in range(m):
        for k,v in zip(["U_t_"+str(j)],[matrix_power(H,j)]):
            U_t[k]=v
    print(len(U_t))
    return U_t

def U(H,t,m):
    U_t={}
    Time=np.arange(0,t,t/m)
    for i in range(m):
        for k,v in zip(["U_t_"+str(i)],[expm((-1j*Time[i]*H))]):
            U_t[k]=v
    return U_t

"""Exact diagonal"""
def Matrix(matrix): #forms the matrix for A+B without and decompostion
    size=int(N-np.sqrt(len(matrix[0])))
    Sum = np.zeros((K,K))
    for i in range(size):
        Sum = Sum + sig(i,size,matrix)
    Sum = Sum + sig(size,size,matrix)
    return Sum
def ED_IS(t,m,bits):
    Psi_0=inital(bits)
    C= J*Matrix(XX)+g*Matrix(Z)
    U_t=U(C,t,m)
    Phi_t= zeros(K,m)
    for i in range(m): #finding phi_t for a time t in steps defined in t=np.arrange(...)
        Phi_t[:,i]=np.matmul(U_t["U_t_"+str(i)],Psi_0)
    return Phi_t
def ED_Heis(t,bits):
    
    Psi_0=inital(bits)
    C= J*(Matrix(XX)+Matrix(YY))
    U_t=U(C,t,M)
    Phi_t= zeros(K,M)
    for i in range(M): #finding phi_t for a time t in steps defined in t=np.arrange(...)
        Phi_t[:,i]=np.matmul(U_t["U_t_"+str(i)],Psi_0)
    return Phi_t

"""Trotter functions"""
def Trot_is(t,m): #perfroms trotter simulation for ising model
    odd,even= Odd_Even(N-1)
    A=form_matrix(Z,0,g,"Z_")
    XX_e= form_matrix(XX, 2,J,"XX_e")
    XX_o= form_matrix(XX, 1,J,"XX_o")
    A_prod=Prod(A,"Z_",t,m,1,range(N))
    B_e=Prod(XX_e,"XX_e",t,m,1,even)
    B_o=Prod(XX_o,"XX_o",t,m,1,odd)
    H=np.matmul(np.matmul(A_prod,B_e),B_o)
    U_t=Steps(H,m)
    Phi_t= zeros(K,m)
    for j in range(m):
        Phi_t[:,j]=matmul(U_t["U_t_"+str(j)],Psi_0)
    Exp_z_t=Exp_z(Phi_t,m)
    return Exp_z_t,Phi_t #retruns Exp_z and statevector Phi for each time step

def Trot_Heis(t,m,bits):
    Psi_0=inital(bits)
    odd,even= Odd_Even(N-1)
    A=form_matrix(Z, -1,0,"Z_") #somthing -1 sign to require use of random h value 
    XX_e=form_matrix(XX, 2,Jx,"XX_") #produces dict of XX even matrices 
    XX_o=form_matrix(XX, 1,Jx,"XX_") #sym
    YY_e=form_matrix(YY, 2,Jy,"YY_") 
    YY_o=form_matrix(YY, 1,Jy,"YY_")
    ZZ_e=form_matrix(ZZ, 2,Jz,"ZZ_")
    ZZ_o=form_matrix(ZZ, 1,Jz,"ZZ_")
    Bx_e=Prod(XX_e,"XX_",t,m,1,even) #products the XX  even index matrices together
    Bx_o=Prod(XX_o,"XX_",t,m,1,odd) #sym
    By_e=Prod(YY_e,"YY_",t,m,1,even)
    By_o=Prod(YY_o,"YY_",t,m,1,odd)
    Bz_e=Prod(ZZ_e,"ZZ_",t,m,1,even)
    Bz_o=Prod(ZZ_o,"ZZ_",t,m,1,odd)
    A_prod=Prod(A,"Z_",t,m,1,range(N)) 
    B_o=matmul(matmul(By_o,Bx_o),Bz_o) #matrix products odd set
    B_e=matmul(matmul(By_e,Bx_e),Bz_e) #matrix products even set
    H=matmul(matmul(A_prod,B_e),B_o) #matrix products all matrices
    print(np.shape(H))
    U_t=Steps(H,m)
    Phi_t=zeros(K,m)
    for j in range(m): #creates state vector for each time step
            Phi_t[:,j]=matmul(U_t["U_t_"+str(j)],Psi_0)
    return Phi_t


"""Shots conversion"""
def Get_Counts(account,Id,backend):
    load_account(account)
    backend = provider.get_backend(backend)
    counts=[]
    with open(Id + ".txt","r")as f:
        data=f.readlines()
    for i in data:
        i.strip()
        job =backend.retrieve_job(i)
        counts.append(job.result().get_counts()) 
    with open(Id + "_Counts.txt","w") as f:
        f.write(json.dumps(counts))
        f.close()
    return counts 

def Open_counts(name):
    with open(name,"r") as f:
        data=f.read()
    counts=json.loads(data)
    m=len(counts)
    shot_array=Counts(counts[::-1],m)[1]
    return shot_array,m
def split(p): #turns bit string into list of integers, only 
    p_list=[]
    p_list=[int(i) for i in p]
    return p_list
def remove(mag): #removes values outside of the hilbert space
    Bin_index=[]
    Bin_not=[]
    Bin_list =[]
    for i in range(K): 
        dum =format(i,"b") #string of binary for ith point 
        l = split(dum) #list of int value 
        l_sum = sum(l)
        if l_sum != mag: #works on total number of 1's in intial state 
            Bin_index.append(i)
        elif l_sum == mag:
            Bin_not.append(i)
            for j in range(N-len(l)):
                l = [0] +l
            Bin_list.append(l)
    return Bin_index,Bin_not, Bin_list 
def Bin(j):
    k = format(j,"b")  
    k_list = split(k)
    for n in range(N-len(k_list)):
        k_list = [0] + k_list
    return k_list #retruns list form of binary rep up to N bits
def Exp_shot(array):
    m = len(array[1])
    exp = np.zeros((m,N))
    
    for l in range(m):
        Phi_t=array[:,l]
        print(Phi_t)
        for j in range(len(Phi_t)): #j is the jth binary state i.e 0 == |00000> for 5 bits
            k_list=Bin(j) 
            for i in range(N):
                if k_list[i]==0:
                    exp[l,i]=exp[l,i]-Phi_t[j]
                else:
                    exp[l,i]=exp[l,i]+Phi_t[j]
    return exp

def Bin_shot(dic,m):
    step_dic={}
    key_list=[]
    for key in dic.keys():
          phi=list(key)[::-1] #flip round list 
          for i in range(N):
              phi[i]=int(phi[i]) #turns key into list of integers 
          B=Binary(phi) #turns into binary number
          key_list.append((B,dic[key])) #appends tuple to key list binary attached to shot count
    return key_list 

def Counts(counts,m): #creates dictonary of binary rep. with shot count 
    shot_array=np.zeros((K,m))
    counts_dic={}
    for i in range(m):
        q=counts[i] #dic of retruned shots at time step i
        for k,v in zip(["step_"+str(i)],[Bin_shot(q,i)]):
            counts_dic[k]=v
    for i in range(m):
        for j in counts_dic["step_"+str(i)]:
            shot_array[j[0],i]=j[1]
    return counts_dic,shot_array

"""Error Mitigation"""

def mitigation(backend,noise): #from qisit measument gate error mitigation
    qr = QuantumRegister(N)
    meas_calibs, state_labels = complete_meas_cal(qr=qr, circlabel='mcal')
    calab_circ = transpile(meas_calibs, backend) #transpiled quantum circuit 
    cal_results = execute(calab_circ, backend=backend,shots=S,optimization_level=2).result() #results of calabration job
    meas_fitter = CompleteMeasFitter(cal_results, state_labels, circlabel='mcal') #correction needed from calibration
    meas_filter = meas_fitter.filter #creates filter to be used on future measurments
    return meas_filter

def meaasument_mitigation(results,job): #check
    qr = QuantumRegister(N)
    state_labels = complete_meas_cal(qr=qr, circlabel='mcal')[1]
    meas_fitter = CompleteMeasFitter(job.result(),state_labels, circlabel='mcal') #correction needed from calibration
    meas_fitter.plot_calibration()
    meas_filter = meas_fitter.filter #creates filter to be used on future measurments
    mit_c=[]
    for i in range(len(results)):
        mit_c.append(meas_filter.apply(results[i].result()).get_counts()) #adjusts measurment based on previously run calabration
    return mit_c, meas_filter.cal_matrix

def remove(mag): #removes values outside of the hilbert space based on total magnitisation
    Bin_index=[]
    Bin_not=[]
    Bin_list =[]
    for i in range(K): 
        dum =format(i,"b") #string of binary for ith point 
        l = split(dum) #list of int value 
        l_sum = sum(l)
        if l_sum != mag: #works on total number of 1's in intial state 
            Bin_index.append(i)
        elif l_sum == mag:
            Bin_not.append(i)
            for j in range(N-len(l)):
                l = [0] +l
            Bin_list.append(l)
    return Bin_index,Bin_not, Bin_list 

def renormalize(remove,array,time_step): #removes values from array outside of hilbert space 
    for i in remove:
        array[i,:]=0
    for j in range(time_step):
        s = sum(array[:,j])
        array[:,j]=array[:,j]/s
    return array

"""Quantum codes"""

def init(C,N,bits): #initalisation circuit
    q_dic={}
    h=bits #bits is the list 
    for j in range(N):
        for k,v in zip(str(j),[h[j]]):
            q_dic[k]=v
        if q_dic[str(j)]==0:
            C.i(j)
        elif q_dic[str(j)]==1:
            C.x(j)
    return C,q_dic
def measure(C): #mesument circuit
    for i in N:
        C.measure(i,)
    return
def RZr(theta,L,C): #somthing in this gate reduces fidelity 
    h_l=h_list[::-1]
    for i in L:
        C.rz(theta*h_l[i],i)
    return C
def RZ(theta,L,C): #Rotation Z gate for each bit 
    for i in L:
        C.rz(theta,i)
    return C
def RXX(theta,L,C):
    for i in L:
        C.rxx(theta,i,i+1)
    return C
def RYY(theta,L,C):
    for i in L:
        C.ryy(theta,i,i+1)
    return C
def RZZ(theta,L,C):
    for i in L:
        C.rzz(theta,i,i+1)
    return C

def Circuit_I(j):
    delta_t=Quan_H.delta_t
    bits=Quan_H.bits
    theta_U=-2*(delta_t)
    thx=-2*Jx*(delta_t)
    circuit = QuantumCircuit(N, name =str(j) +"th step") 
    circuit=init(circuit,N,bits)[0] #initalize state 
    odd,even = Odd_Even(N-1)
    odd=odd[::-1] #flip diection of list
    even=even[::-1]
    for i in range(j):
        circuit = RXX(thx,even,circuit) 
        circuit = RXX(thx,odd,circuit)
        circuit = RZ(theta_U,q_i,circuit)
    return circuit

def Circuit_sym_H(j): #Symmetric implementation 
    delta_t=Quan_H.delta_t
    bits=Quan_H.bits
    theta_U=-2*(delta_t)
    thx=-2*Jx*(delta_t)
    thy=-2*Jy*(delta_t)
    thz=-2*Jz*(delta_t)
    circuit = QuantumCircuit(N, name =str(j) +"th step") 
    circuit=init(circuit,N,bits)[0] #initalize state 
    odd,even = Odd_Even(N-1)
    odd=odd[::-1] #flip diection of list
    even=even[::-1]
    for i in range(j):
        circuit = RZr(theta_U/2,q_i,circuit)
        circuit = RYY(thy/2,even,circuit)
        circuit = RXX(thx/2,even,circuit) 
        circuit = RZZ(thz/2,even,circuit)
        circuit = RYY(thy,odd,circuit)
        circuit = RXX(thx,odd,circuit)
        circuit = RZZ(thz,odd,circuit)
        circuit = RYY(thy/2,even,circuit)
        circuit = RXX(thx/2,even,circuit) 
        circuit = RZZ(thz/2,even,circuit)
        circuit = RZr(theta_U/2,q_i,circuit)
    return circuit

def Circuit_H(j): #for jth trotter step perfroms j steps of the circuit 
    delta_t=Quan_H.delta_t
    bits=Quan_H.bits
    theta_U=-2*(delta_t)
    thx=-2*Jx*(delta_t)
    thy=-2*Jy*(delta_t)
    thz=-2*Jz*(delta_t)
    circuit = QuantumCircuit(N, name =str(j) +"th step") 
    circuit=init(circuit,N,bits)[0] #initalize state 
    odd,even = Odd_Even(N-1)
    odd=odd[::-1] #flip diection of list
    even=even[::-1]
    for i in range(j):
        circuit = RZr(theta_U,q_i,circuit)
        circuit = RYY(thy,even,circuit)
        circuit = RXX(thx,even,circuit)
        if thz !=0: #removes this gate if RZZ rotation not occuring 
            circuit = RZZ(thz,even,circuit)
        else:
            pass
        circuit = RYY(thy,odd,circuit)
        circuit = RXX(thx,odd,circuit)
        if thz !=0: #removes this gate if RZZ not occuring
            circuit = RZZ(thz,odd,circuit)
        else:
            pass
    return circuit


"""Run experiment"""
def Run(Sys,Back,meas):
    Sys.circuit.measure(meas, meas) #measures q_i in circuit 
    compiled_circuit = transpile(Sys.circuit,backend=Back) 
    job = Back.run(compiled_circuit,shots=S)
    result = job.result()
    return result

def Quan_H(t=1,m=5,bits=[1,1,1,0,0],Back=QasmSimulator(),Circuit=Circuit_H,s_v=None,noise=NoiseModel(),meas=q_i):
    Quan_H.delta_t=t/m
    n=len(bits)
    Quan_H.bits=bits
    Q_array=zeros(2**n,m)
    odd,even = Odd_Even(n-1)
    counts = []
    simulator= QasmSimulator()
    for j in range(m): #running circuit for each trotter step
        Quan_H.circuit=circuit = Circuit(j) #circuit formed upto jth s
         #creates noiseless statevector from that point in the circuit
        if s_v==None:
            result= Run(Quan_H,Back,q_i)#runs circuit with noise
            counts.append(result.get_counts())
        elif s_v==True:
          circuit.save_statevector()
          result = simulator.run(circuit).result()
          Q_array[:,j] = result.get_statevector(circuit, decimals=10)
    if s_v==True:
        return Q_array
    elif s_v==None:
        return counts
    
"""plotting"""
def colour_plot(Array,t=1, title="test"):
    fig_T, ax_T = plt.subplots()
    im_T=ax_T.imshow(np.real(Array), origin="lower",extent=(0-0.5, (N-1)+0.5, 0, t*10), cmap="jet", vmin=1,vmax=-1, aspect="auto")
    plt.title(title)
    plt.xlabel("Bits")
    plt.ylabel("time/ms")
    fig_T.colorbar(im_T)
    return 

