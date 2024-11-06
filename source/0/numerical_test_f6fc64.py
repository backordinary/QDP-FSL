# https://github.com/XXChain/Research-Project-/blob/36e96be5dc32e979d358a738515a6f3c5a1eaa8d/Numerical_test.py
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 15:27:53 2022

@author: User
"""
import numpy as np
import json
from scipy.linalg import expm
from qiskit import QuantumCircuit,QuantumRegister, Aer, assemble,execute,transpile,schedule
from qiskit.providers.ibmq import least_busy
from qiskit.ignis.mitigation.measurement import complete_meas_cal, CompleteMeasFitter
from qiskit.providers.aer import QasmSimulator
import matplotlib.pyplot as plt 
import random
from mitiq.zne.scaling import fold_gates_at_random,fold_global
from mitiq.zne.scaling.folding import fold_all
class Pauli:
    Z= np.zeros((2,2))
    X=np.zeros((2,2))
    Y=np.zeros((2,2),dtype=np.complex128)
    Z[0,0] = 1
    Z[1,1] = -1
    X[0,1] = 1
    X[1,0] = 1
    Y[0,1]=-1j
    Y[1,0]=1j
    YY= np.kron(Y,Y)
    ZZ = np.kron(Z,Z)
    XX = np.kron(X,X)

class Funcs:
    def shannon(array,):
        array=np.where(array<=0e-5,1,array)
        h=np.sum(array*np.log2(array),axis=0)
        return -h
    def vonNeum(array):
        array=np.where(array<=0e-4,1,array)
        h=np.sum(array*np.log(array),axis=0)
        return -h       
    def cubic(x,a,b,c,d):
        return a*(x**3) +b*(x**2) +c*x + d
        
    def RSS(y,f_x): #look into more of this 
        return np.sum((y-f_x)**2,axis=0)
    def Damped(x,a,b,c,d):
        return c*np.exp(-b*x)*np.cos(a*x+d)
    
    def inital(self): #intital state 
        f_s=Funcs.Binary(self.bits)
        Psi_0=np.zeros(self.K)
        Psi_0[f_s]=1
        return Psi_0
    
    def Binary(bits): #converts bits ofr intital state into a binary number
        D_num=0
        for i in range(len(bits)):
            D_num= D_num + bits[i]*(2**(i))
        return D_num
    def Exp(u,v): #finds expectation values <v|u|v>
        assert(len(np.shape(u))==2) #check is square 
        #assert(len(v)==len(u[0,:])) #check same no of values
        return np.dot(-np.conj(v),np.matmul(u,v)) 
    
    def sig(n,N,sig): #finding matirx without using for loop
        A=np.identity(int(2**(n)))
        B=np.identity(int(2**(N-n)))
        return np.kron(A,np.kron(sig,B))
    
    def Odd_Even(N):
        odd=[]
        even=[]
        for i in range(N):
            if i%2==0:
                even.append(i)
            else:
                odd.append(i)
        return odd, even  
        
    def Matrix(self,sigma): #forms the matrix for A+B without and decompostion
        size=int(self.N-np.sqrt(len(sigma[0])))
        Sum = np.zeros((self.K,self.K))
        for i in range(size):
            Sum = Sum + Funcs.sig(i,size,sigma)
        Sum = Sum + Funcs.sig(size,size,sigma)
        return Sum

    def form_matrix(N,sigma,index): #forms dictionary of decomposed matrices
        size=int(N-np.log2(len(sigma[0])))
        if index==None:
            index=np.arange(0,size,1)
        else:
            index=index
        List=[]
        for i in index:
            List.append(Funcs.sig(i,size,sigma))
        return List
    
    def Prod(self,A,J): #products e^-J*Gate(i)*(delta_t*c) c being 1 or 1/2 
        A_prod=np.identity(int(2**self.N)) 
        for i in A: #steps being odd list even list or list of range N 
            A_prod=np.matmul(expm(-1j*i*self.delta_t*J),A_prod) 
        return A_prod
    
    def Phi_t(self,H): #finds Phi_t for M steps 
        Phi_t= np.zeros((self.K,self.M+1),dtype=np.complex128)
        for i in range(self.M):
            Phi_t[:,i]=np.matmul(np.linalg.matrix_power(H,i),self.Psi_0)
        Phi_t[:,self.M]=np.matmul(np.linalg.matrix_power(H,self.M),self.Psi_0)
        return Phi_t
    
    def Bin_shot(dic,m):
        key_list=[]
        for key in dic.keys():
              phi=list(key)[::-1] #flip round list 
              for i in range(len(phi)):
                  phi[i]=int(phi[i]) #turns key into list of integers 
              B=Funcs.Binary(phi) #turns into binary number
              key_list.append((B,dic[key])) #appends tuple to key list binary attached to shot count
        return key_list 
    
    def Bin(j,N): #turns integer into binary list of length N
        k = format(j,"b") #
        k_list = Funcs.split(k)
        for n in range(N-len(k_list)):
            k_list = [0] + k_list
        return k_list
    
    def split(p): #turns bit string into list of integers
        p_list=[int(i) for i in p]
        return p_list

    
    def Cts_Sarray(counts): #creates dictonary of binary rep. with shot count 
         #number of experiments/timesteps
        m=len(counts)
        K=2**(len(list(counts[0].keys())[0]))
        shot_array=np.zeros((K,m))
        counts_dic={}
        for i in range(m):
            q=counts[i] #dic of retruned shots at time step i
            for k,v in zip(["step_"+str(i)],[Funcs.Bin_shot(q,i)]):
                counts_dic[k]=v
        for i in range(m):
            for j in counts_dic["step_"+str(i)]:
                shot_array[j[0],i]=j[1]
        return shot_array
    

class Circuits:
    
    def init(bits,C): #initalisation circuit
        for j in range(len(bits)):
            k =bits[j]
            if k==0:
                C.i(j)
            elif k==1:
                C.x(j)
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
    def IRXX(theta,L,C):
        for i in L:
            C.rxx.inverse(theta,i,i+1)
        return C
    def IRYY(theta,L,C):
        for i in L:
            C.ryy.inverse(theta,i,i+1)
        return C
    def IRZZ(theta,L,C):
        for i in L:
            C.rzz.inverse(theta,i,i+1)
        return C
    
    def Fold_rand(circ, s_f):
        return fold_gates_at_random(circ, scale_factor=s_f,fidelities={"single": 1.0,
                                       "CNOT": 0.99})
    
    def Fold_golbal(circ, s_f):
        return fold_global(circ, scale_factor=s_f)
    
    def Fold_all(circ,s_f):
        return fold_all(circ,scale_factor=s_f,fidelities={"single": 1.0,
                                       "CNOT": 0.99})
    
    
    def Circuit_I(self,j): 
        bits=self.bits 
        delta_t=self.delta_t
        theta_U=-2*self.g*(delta_t)
        thx=-2*self.J*(delta_t)
        N=self.N
        q_i=self.meas
        circuit = QuantumCircuit(N,N, name =str(j) +"th step") 
        circuit=Circuits.init(self.bits,circuit) #initalize state 
        odd,even = Funcs.Odd_Even(N-1)
        odd=odd[::-1] #flip diection of list
        even=even[::-1]
        for i in range(j):
            circuit = Circuits.RXX(thx,even,circuit) 
            circuit = Circuits.RXX(thx,odd,circuit)
            circuit = Circuits.RZ(theta_U,q_i,circuit)
        return circuit
    
    def Circuit_sym_H(self,j): #Symmetric implementation jth run
        delta_t=self.delta_t
        theta_U=-2*self.g*(delta_t)
        thx=-2*self.J*(delta_t)
        thy=-2*self.J*(delta_t)
        thz=-2*self.Jz*(delta_t)
        N=self.N
        circuit = QuantumCircuit(N,N, name =str(j) +"th step") 
        circuit=Circuits.init(self.bits,circuit) #initalize state 
        odd,even = Funcs.Odd_Even(N-1)
        odd=odd[::-1] #flip diection of list
        even=even[::-1]
        for i in range(j):
            circuit = Circuits.RYY(thy/2,even,circuit)
            circuit = Circuits.RXX(thx/2,even,circuit) 
            circuit = Circuits.RZZ(thz/2,even,circuit)
            circuit = Circuits.RYY(thy,odd,circuit)
            circuit = Circuits.RXX(thx,odd,circuit)
            circuit = Circuits.RZZ(thz,odd,circuit)
            circuit = Circuits.RYY(thy/2,even,circuit)
            circuit = Circuits.RXX(thx/2,even,circuit) 
            circuit = Circuits.RZZ(thz/2,even,circuit)
        return circuit
    
    def Circuit_H(self,j): #for jth trotter step perfroms j steps of the circuit
        delta_t=self.delta_t
        thx=-2*self.J*(delta_t)
        thy=-2*self.J*(delta_t)
        thz=-2*self.Jz*(delta_t)
        N=self.N
        circuit = QuantumCircuit(N,N, name =str(j) +"th step") 
        circuit=Circuits.init(self.bits,circuit) #initalize state 
        odd,even = Funcs.Odd_Even(N-1)
        odd=odd[::-1] #flip diection of list
        even=even[::-1]
        for i in range(j):
            circuit = Circuits.RYY(thy,even,circuit)
            circuit = Circuits.RXX(thx,even,circuit)
            if thz !=0.0: #removes this gate if RZZ rotation not occuring 
                circuit = Circuits.RZZ(thz,even,circuit)
            else:
                pass
            circuit = Circuits.RYY(thy,odd,circuit)
            circuit = Circuits.RXX(thx,odd,circuit)
            if thz !=0.0: #removes this gate if RZZ rotation not occuring 
                circuit = Circuits.RZZ(thz,odd,circuit)
            else:
                pass
        return circuit
    
    def Inverse_H(self,j): #for jth trotter step perfroms j steps of the circuit
        delta_t=self.delta_t
        thx=-2*self.J*(delta_t)
        thy=-2*self.J*(delta_t)
        thz=-2*self.Jz*(delta_t)
        N=self.N
        circuit = QuantumCircuit(N,N, name =str(j) +"th step") 
        circuit=Circuits.init(self.bits,circuit) #initalize state 
        odd,even = Funcs.Odd_Even(N-1)
        odd=odd[::-1] #flip diection of list
        even=even[::-1]
        for i in range(j):
            circuit = Circuits.IRYY(thy,even,circuit)
            circuit = Circuits.IRXX(thx,even,circuit)
            if thz !=0.0: #removes this gate if RZZ rotation not occuring 
                circuit = Circuits.IRZZ(thz,even,circuit)
            else:
                pass
            circuit = Circuits.IRYY(thy,odd,circuit)
            circuit = Circuits.IRXX(thx,odd,circuit)
            if thz !=0.0: #removes this gate if RZZ rotation not occuring 
                circuit = Circuits.IRZZ(thz,odd,circuit)
            else:
                pass
        return circuit

class Experiment:
    def __init__(self,name="Test", meas=None, M=2,bits=[1,0,1,0,1], Time=1,J=-1,Jz=0,g=0, op=2, \
                 Shots=20000,model="H",ideal=False, folds=1,backend=QasmSimulator(),state_labels=None,cal_ID=None):
        self.M=M
        self.bits=bits
        self.name=name
        self.Time= Time
        self.delta_t=delta_t=Time/M
        self.N=len(bits)
        self.K= int(2**self.N)
        self.Psi_0=Funcs.inital(self)
        self.Mag=sum(bits)
        self.J=J
        self.Jz=Jz
        self.g=g
        self.Shots=Shots
        self.ideal=ideal
        self.op=op
        self.backend=backend
        self.cal_results=None 
        self.folds=folds
        self.q_i=q_i=np.arange(0,len(bits),1)
        if meas==None:
            self.meas=q_i
        else:
            self.meas=meas
        self.model=model
        self.state_labels=state_labels
        self.cal_ID=cal_ID
        self.Dic={"name":name, "bits":bits,"M":M,"Time":Time,"folds":folds,"J":J,"Jz":Jz,"g":g,"Shots":Shots,"backend":backend.name() \
                  , "cal_ID":cal_ID,"state_labels":state_labels}


class Error_mit:
        
    def calibration(self): #from qisit measument gate error mitigation
        meas_calibs, state_labels = complete_meas_cal(qr=QuantumRegister(self.N), circlabel='mcal')
        calab_circ = transpile(meas_calibs, self.backend) #transpiled quantum circuit 
        cal_results = execute(calab_circ, backend=self.backend,shots=self.Shots,optimization_level=2).result() #results of calabration job
        self.cal_results =cal_results 
        self.Dic["cal_ID"]=cal_ID=cal_results.job_id
        self.Dic["state_labels"]=state_labels
        self.state_labels = state_labels
        backend_name=self.backend.name()
        with open(backend_name+ "_cal_results"+".txt","w") as f:
            f.write(cal_results.job_id)
            f.close()
        with open(backend_name+ "_state_labels"+".txt","w") as f:
            json.dump(state_labels,f)
            f.close()   
        return cal_ID, state_labels
    
    def measument_mitigation(obj): #check
        meas_fitter = CompleteMeasFitter(obj.cal_results,obj.state_labels, circlabel='mcal') #correction needed from calibratio
        meas_filter = meas_fitter.filter #creates filter to be used on future measurments
        mit_c=meas_filter.apply(obj.result).get_counts() #applies measurment mitigation result 
        meas_fitter.plot_calibration()
        obj.mit_c=mit_c
        obj.mit_s=Funcs.Cts_Sarray(mit_c)/obj.Shots 
        obj.fit=meas_filter
        obj.cal_matrix= meas_filter.cal_matrix
        return mit_c
    
    def remove(mag,K): #returns lits of states to remove based on total magnitistion
        Bin_index=[]
        Bin_not=[]
        Bin_list =[]
        N=int(np.log2(K))
        for i in range(K): 
            dum =format(i,"b") #string of binary for ith point 
            l = Funcs.split(dum) #list of int value 
            l_sum = sum(l)
            if l_sum != mag: #works on total number of 1's in intial state 
                Bin_index.append(i)
            elif l_sum == mag:
                Bin_not.append(i)
                for j in range(N-len(l)):
                    l = [0] +l
                Bin_list.append(l)
        return Bin_index,Bin_not
    
    def renormalize(array,Sz): #removes values from array outside of hilbert space 
        K=int(len(array[:,0]))
        m=len(array[1])
        array2=array
        remove=Error_mit.remove(Sz,K)[0]
        array2[remove,:]=0
        for j in range(m):
            s = sum(array2[:,j])
            array2[:,j]=array2[:,j]/s
        return array2
   

class Backends:
    from qiskit import IBMQ 
    account="FOO"
    IBMQ.save_account(account)
    IBMQ.load_account()
    provider = IBMQ.get_provider('ibm-q')
    def least_busy():
        small_devices = Backends.provider.backends(filters=lambda x: x.configuration().n_qubits == 5
                                    and not x.configuration().simulator)
    
        backendQ=least_busy(small_devices)
        print(backendQ)
        return backendQ
    
    def get_backend(name):
        return Backends.provider.get_backend(name)
    
    def noise_model(name):
        from qiskit.providers.aer.noise import NoiseModel
        provider = Backends.provider
        backend = provider.get_backend(name)
        noise_model = NoiseModel.from_backend(backend)
        return noise_model
    
    def get_noise(p): #code from Qiskit noise models 
        from qiskit.providers.aer.noise import NoiseModel
        from qiskit.providers.aer.noise.errors import pauli_error
        error_meas = pauli_error([('X',p), ('I', 1 - p)])
    
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(error_meas, "measure") # measurement error is applied to measurements
        return noise_model
        
        
class Numerical(Experiment):    
    def ED(self):
        H = self.J*(Funcs.Matrix(self,Pauli.XX)+Funcs.Matrix(self,Pauli.YY))+ self.Jz*(Funcs.Matrix(self, Pauli.ZZ))
                                                                    # #Hesienburg ED matrix
        self.Phi_ED = Phi_ED= Funcs.Phi_t(self,expm(-1j*H*self.delta_t))
        self.Exp_ED=Observables.Exp_z(Phi_ED)
        np.savetxt(self.name + "_ED.txt",Phi_ED)
        return Phi_ED
    
    def Trot(self):
        N=self.N
        odd,even= Funcs.Odd_Even(N-1)
        A=(Funcs.form_matrix(N,Pauli.Z,None)) #somthing -1 sign to require use of random h value 
        XX_e=(Funcs.form_matrix(N,Pauli.XX,even)) #produces dict of XX even matrices 
        XX_o=(Funcs.form_matrix(N,Pauli.XX,odd))
        YY_e=(Funcs.form_matrix(N,Pauli.YY,even))
        YY_o=(Funcs.form_matrix(N,Pauli.YY,odd))
        ZZ_e=(Funcs.form_matrix(N,Pauli.ZZ,even))
        ZZ_o=(Funcs.form_matrix(N,Pauli.ZZ,odd))
        Bx_e=Funcs.Prod(self,XX_e,self.J) #products the XX  even index matrices together
        Bx_o=Funcs.Prod(self,XX_o,self.J) #sym
        By_e=Funcs.Prod(self,YY_e,self.J) 
        By_o=Funcs.Prod(self,YY_o,self.J) 
        Bz_e=Funcs.Prod(self,ZZ_e,self.Jz) 
        Bz_o=Funcs.Prod(self,ZZ_o,self.Jz)
        A_prod=Funcs.Prod(self,A,self.g)
        B_o=np.matmul(np.matmul(By_o,Bx_o),Bz_o) #matrix products odd set
        B_e=np.matmul(np.matmul(By_e,Bx_e),Bz_e) #matrix products even set
        H=np.matmul(B_e,B_o) #matrix products all matrices
        self.Phi_T=Phi_T=Funcs.Phi_t(self,H)
        self.Exp_T=Observables.Exp_z(Phi_T)
        np.savetxt(self.name + "_Trt.txt",Phi_T)
        return self.Phi_T
    
    def Trot_sym(self):
        N=self.N
        odd,even= Funcs.Odd_Even(N-1)
        XX_e=(Funcs.form_matrix(N,Pauli.XX,even)) #produces dict of XX even matrices 
        XX_o=(Funcs.form_matrix(N,Pauli.XX,odd))
        YY_e=(Funcs.form_matrix(N,Pauli.YY,even))
        YY_o=(Funcs.form_matrix(N,Pauli.YY,odd))
        ZZ_e=(Funcs.form_matrix(N,Pauli.ZZ,even))
        ZZ_o=(Funcs.form_matrix(N,Pauli.ZZ,odd))
        Bx_e=Funcs.Prod(self,XX_e,self.J/2) #products the XX  even index matrices together
        Bx_o=Funcs.Prod(self,XX_o,self.J) #sym
        By_e=Funcs.Prod(self,YY_e,self.J/2) 
        By_o=Funcs.Prod(self,YY_o,self.J) 
        Bz_e=Funcs.Prod(self,ZZ_e,self.Jz/2) 
        Bz_o=Funcs.Prod(self,ZZ_o,self.Jz)
        B_o=np.matmul(np.matmul(By_o,Bx_o),Bz_o) #matrix products odd set
        B_e=np.matmul(np.matmul(By_e,Bx_e),Bz_e) #matrix products even set
        H=np.matmul(np.matmul(B_e,B_o),B_e)
        self.Phi_T=Phi_T=Funcs.Phi_t(self,H)
        self.Exp_T=Observables.Exp_z(Phi_T)
        np.savetxt(self.name + "_SymTrt.txt",Phi_T)
        
        pass

class Quantum(Experiment): 
    
        def State_Vector(self): 
            Q_array=np.zeros((self.K,self.M+1),dtype=np.complex128)
            simulator= QasmSimulator()
            for j in range(self.M+1): #running circuit for each trotter step
                if self.model=="sym":
                    circuit = Circuits.Circuit_sym_H(self,j) 
                elif self.model=="H":
                    circuit = Circuits.Circuit_H(self, j)
                elif self.model =="I":
                    circuit = Circuits.Circuit_I(self,j)
                circuit.save_statevector()
                result = simulator.run(circuit).result()
                Q_array[:,j] = result.get_statevector(circuit, decimals=10)
            self.Q_array=Q_array
            self.Exp_Z=Observables.Exp_z(Q_array)
            return self.Q_array
        
        def Quan_H(self):
            M=self.M
            delta_t=self.delta_t
            S=self.Shots
            folds=self.folds
            meas=self.meas
            if folds ==1:
                op=2
            else:
                op=self.op
            Circ_list=[]
            for j in range(M+1): #running circuit for each trotter steps
                if self.model=="sym":
                    circuit = Circuits.Circuit_sym_H(self,j) 
                elif self.model=="H":
                    circuit = Circuits.Circuit_H(self, j)
                elif self.model =="I":
                    circuit = Circuits.Circuit_I(self,j)
                circuit=Circuits.Fold_rand(circuit.decompose(),s_f=folds)
                circuit.measure(meas,meas)
                Circ_list.append(circuit)
            job=execute(Circ_list,self.backend, basis_gates=['id', 'rz', 'sx', 'x', 'cx'], \
                                                                optimization_level=op,shots=S)
            self.JobID = JobID= job.job_id()
            self.Dic["JobID"]=JobID
            FileManage.save_json(self)
            # with open(self.name+ "_result"+".txt","w") as f:
            #     f.write(job.job_id())
            #     f.close()
            # print(job.status)
            # result=job.result()
            # counts=result.get_counts()
            # self.JobID = JobID= job.job_id()
            # self.counts=counts
            # self.shot_array=Funcs.Cts_Sarray(counts)/S
            # self.result=result
            # self.Dic["JobID"]=JobID
            # FileManage.save_json(self)
            return self

class Observables:
    
    def fidelity(state_vector,shots_array): #tests fidelity between state_vector probs and shots array probs
        assert np.shape(state_vector)==np.shape(shots_array)
        probs=np.abs(state_vector)**2
        return np.abs(np.linalg.norm(probs-shots_array,axis=0))
    
    def Exp_shot(array):
        N= int(np.log2(len(array[:,0])))
        m = int(len(array[0,:]))
        exp = np.zeros((m,N))
        for l in range(m):
            Phi_t=array[:,l]
            for j in range(len(Phi_t)): #j is the jth binary state i.e 0 == |00000> for 5 bits
                k_list=Funcs.Bin(j,N) 
                for i in range(N):
                    if k_list[i]==0:
                        exp[l,i]=exp[l,i]-Phi_t[j]
                    else:
                        exp[l,i]=exp[l,i]+Phi_t[j]
        return exp
    
    def Exp_z(array): #finding sig_Z values at each value of |phi(t)>
        m=int(len(array[0,:]))
        N=int(np.log2(len(array[:,0])))
        Exp_z_t=np.zeros((m,N),dtype=np.complex128)
        for i in range(N):  
            q=np.empty((m,),dtype=np.complex128) 
            for j in range(m):
                q[j]=Funcs.Exp(Funcs.sig(i,N-1,Pauli.Z),array[:,j]) 
            Exp_z_t[:,i]=q 
        return Exp_z_t
    
    def single(Exp,bit):
        return Exp[bit,:]
    
    def Shannon(array):
        array=np.where(array<=0e-6,1,array)
        h=np.sum(array*np.log2(array),axis=0)
        return -h
    
    def VonNeum(array):
        array=np.where(array<=0e-6,1,array)
        h=np.sum(array*np.log(array),axis=0)
        return -h
    
    def RSS():
        pass 

class FileManage:
    
    def Save_Counts(self,name=None): #Saves Experiment to arrays 
        if name !=None:
            title=name
        else:
            if self.name !=None:
                title=self.name
            elif self.name==None:
                title=self.JobID
        with open(title + "_Counts.txt","w") as f:
            f.write(json.dumps(self.counts))
            f.close()
        return
        
    def save_json(self):
        with open(self.name + "_JobDic.txt","w") as f:
            f.write(json.dumps(self.Dic))
            f.close()
        return
        
    def open_json(name):
        with open(name,"r") as f:
            dum=f.read()
            Dic=json.loads(dum)
            f.close()
        return Dic

class Results: 
    def __init__(self,Dic):
        self.bits=bits=Dic["bits"]
        self.M=M=Dic["M"]
        self.Time=Time=Dic["Time"]
        self.delta_t=delta_t=Time/M
        self.N=len(bits)
        self.K= int(2**self.N)
        self.Mag=sum(bits)
        self.J=Dic["J"]
        self.Jz=Dic["Jz"]
        self.g=Dic["g"]
        self.Shots=Dic["Shots"]
        self.backend_name=Dic["backend"]
        self.folds=Dic["folds"]
        self.JobID=Dic["JobID"]
        self.name=Dic["name"]
        self.cal_ID=Dic["cal_ID"]
        self.state_labels=Dic["state_labels"]
        
    def load_results(self):
            provider=Backends.provider 
            print(self.backend_name)
            self.backend=backend=provider.get_backend(self.backend_name)
            job=backend.retrieve_job(self.JobID)
            self.result=result=job.result()
            self.counts=result.get_counts()
            self.shot_array=Funcs.Cts_Sarray(self.counts)/self.Shots
            self.cal_results=backend.retrieve_job(self.cal_ID).result()
            Error_mit.measument_mitigation(self)
            self.mit_c=Error_mit.measument_mitigation(self)
            self.norm_s=Error_mit.renormalize(self.shot_array, self.Mag)
            self.full_mit=Error_mit.renormalize(self.mit_s, self.Mag)
            self.Exp_MeasMit=Observables.Exp_shot(self.mit_s)
            self.Exp_FullMit=Observables.Exp_shot(self.full_mit)
            self.Exp_NormMit=Observables.Exp_shot(self.norm_s)
            self.Exp_NoMit=Observables.Exp_shot(self.shot_array)


class Plots: 
    def site_mag(obj,bit=0,ED=False,Trot=False,s=True,nom=True,Max=True):
        Title="Exp_Z of  site " +  str(bit) + \
            "\n"+"Experiment : " +obj.name
        m1=len(obj.Exp_NoMit[:,0])
        print(m1)
        T=obj.Time
        x=np.arange(0,T+T/m1,(T+T/m1)/m1)
        print(len(x))
        if ED==True:
            m=40
            T=(obj.M)*obj.delta_t
            x_ED= np.arange(0,T+T/m,(T+T/m)/(m+1))
            ED=Numerical(name="steps_40",M=m,Time=obj.Time,bits=obj.bits)
            Exp_ED=Observables.Exp_z(ED.ED())
            plt.plot(x_ED,np.real(Exp_ED[:,bit]), c="black",label="ED")
        elif ED==False:
            pass
        if Trot==True:
            Trot=Numerical(name="trotter_"+str(obj.M), M=obj.M,Time=obj.Time,bits=obj.bits)
            Exp_T=Observables.Exp_z(Trot.Trot())
            plt.plot(x,np.real(Exp_T[:,bit]),c="green",marker=".",label="Trot")
        elif ED==False:
            pass
        if nom==True:
            plt.plot(x,obj.Exp_NoMit[:,bit],c="blue",marker="v",label="Original")
        else:
            pass
        if nom==True:
            plt.plot(x,obj.Exp_NormMit[:,bit],c="orange",marker="v",label="Normalised")
        else:
            pass
        if Max==True:
            plt.plot(x,obj.Exp_FullMit[:,bit],c="grey",marker="v",label="Max mit")
        else:
            pass
        plt.grid()
        plt.title(Title)
        plt.xlabel("Time Jt")
        plt.ylabel("Mag Site  " + str(bit))
        plt.legend(loc="best",fontsize='small')
        plt.savefig("Site "+str(bit) +obj.name +".png")
        plt.show()
        
    def BMS(Title,B,M,S,bit,ED=False,Trot=False):
        Title=Title  + "site " +str(bit)
        m1=len(B.Exp_NoMit[:,0])
        print(m1)
        T=B.Time
        Bits=B.bits
        x=np.arange(0,T+T/m1,(T+T/m1)/m1)
        print(len(x))
        if ED==True:
            m=40
            x_ED= np.arange(0,T+T/m,(T+T/m)/(m+1))
            ED=Numerical(name="steps_40",M=m,Time=T,bits=Bits)
            Exp_ED=Observables.Exp_z(ED.ED())
            plt.plot(x_ED,np.real(Exp_ED[:,bit]), c="black",label="ED")
        elif ED==False:
            pass
        if Trot==True:
            Trot=Numerical(name="trotter_"+str(B.M), M=B.M,Time=T,bits=Bits)
            Exp_T=Observables.Exp_z(Trot.Trot())
            plt.plot(x,np.real(Exp_T[:,bit]),c="green",marker=".",label="Trot")
        elif Trot==False:
            pass
        if B!=None:
            plt.plot(x,B.Exp_FullMit[:,bit],c="blue",marker="v",label=B.name)
        else:
            pass
        if M!=None:
            plt.plot(x,M.Exp_FullMit[:,bit],c="orange",marker="v",label=M.name)
        else:
            pass
        if S!=None:
            plt.plot(x,S.Exp_FullMit[:,bit],c="grey",marker="v",label=S.name)
        else:
            pass
        plt.grid()
        plt.title(Title)
        plt.xlabel("Time Jt")
        plt.ylabel('Local Magnitization')
        plt.legend(loc="best",fontsize='small')
        plt.savefig(Title+str(B.M)+" Site "+str(bit)+".png")
        plt.show()
        if Trot==True:
            return Trot
        else: 
            pass

    def Mit(Title,obj,bit=2,F=True,Norm=True,Meas=True,No=True,ED=False,Trot=False):
        Title=Title  + "site " +str(bit)
        m1=len(obj.Exp_NoMit[:,0])
        print(m1)
        T=obj.Time
        Bits=obj.bits
        x=np.arange(0,T+T/m1,(T+T/m1)/m1)
        print(len(x))
        if ED==True:
            m=40
            x_ED= np.arange(0,T+T/m,(T+T/m)/(m+1))
            ED=Numerical(name="steps_40",M=m,Time=T,bits=Bits)
            Exp_ED=Observables.Exp_z(ED.ED())
            plt.plot(x_ED,np.real(Exp_ED[:,bit]), c="black",label="ED")
        elif ED==False:
            pass
        if Trot==True:
            Trot=Numerical(name="trotter_"+str(obj.M), M=obj.M,Time=T,bits=Bits)
            Exp_T=Observables.Exp_z(Trot.Trot())
            plt.plot(x,np.real(Exp_T[:,bit]),c="green",marker=".",label="Trot")
        elif Trot==False:
            pass
        if F==True:
            plt.plot(x,obj.Exp_FullMit[:,bit],c="blue",marker="v",label="Full_Mit")
        else:
            pass
        if Meas==True:
            plt.plot(x,obj.Exp_MeasMit[:,bit],c="orange",marker="v",label="Meas_Mit")
        else:
            pass
        if Norm==True:
             plt.plot(x,obj.Exp_NormMit[:,bit],c="grey",marker="v",label="Norm_Mit")
        else:
            pass
        if No==True:
             plt.plot(x,obj.Exp_NoMit[:,bit],marker="v",label="No_Mit")
        else:
            pass
        plt.grid()
        plt.title(Title)
        plt.xlabel("Time Jt")
        plt.ylabel('Local Magnitization')
        plt.legend(loc="best",fontsize='small')
        plt.savefig(Title+str(obj.M)+" Site "+str(bit)+".png")
        plt.show()
        if Trot==True:
            return Trot
        else: 
            pass
    def Bigplot(Data,bit=0,ED=False,Trot=False):
        B=Data[0]
        Title="Exp_Z of  site " +  str(bit)
        m1=len(B.Exp_NoMit[:,0])
        print(m1)
        T=B.Time
        Bits=B.bits
        x=np.arange(0,T+T/m1,(T+T/m1)/m1)
        print(len(x))
        if ED==True:
            m=40
            x_ED= np.arange(0,T+T/m,(T+T/m)/(m+1))
            ED=Numerical(name="steps_40",M=m,Time=T,bits=Bits)
            Exp_ED=Observables.Exp_z(ED.ED())
            plt.plot(x_ED,np.real(Exp_ED[:,bit]), c="black",label="ED")
        elif ED==False:
            pass
        if Trot==True:
            Trot=Numerical(name="trotter_"+str(B.M), M=B.M,Time=T,bits=Bits)
            Exp_T=Observables.Exp_z(Trot.Trot())
            plt.plot(x,np.real(Exp_T[:,bit]),c="green",marker=".",label="Trot")
        elif Trot==False:
            pass
        for i in Data:
            plt.plot(x,Data.Exp_FullMit[:,bit],marker="v",label=Data.name)
        plt.grid()
        plt.title(Title)
        plt.xlabel("Time Jt")
        plt.ylabel('Local Magnitization')
        plt.legend(loc="best",fontsize='small')
        plt.savefig("BMS Step"+str(B.M)+" Site "+str(bit)+".png")
        plt.show()
        if Trot==True:
            return Trot
        else: 
            pass
        
            
        
    def colour_plot(obj,title,Exp,Time=1):
        N=obj.N
        x_bar=np.arange(N)
        if obj==None:
            T=Time
        else:
            T=obj.Time
        Title=title
        fig_T, ax_T = plt.subplots()
        im_T=ax_T.imshow(np.real(Exp), origin="lower",extent=(0-0.5, (N-1)+0.5, 0, T*10), cmap="jet", vmin=1,vmax=-1, aspect="auto")
        ax_T.set_xticks(x_bar,)
        plt.title(title)
        plt.xlabel("Bits")
        plt.ylabel("Time/ms")
        fig_T.colorbar(im_T)
        plt.savefig(Title+"colourplot.png")
        return 


    