# https://github.com/fcarnazza/EntangleDynamics/blob/8312ff6d85e648fcce7d5aecd7a5d250c9c87028/two_qubits/utils.py
# subroutines

from qiskit import *
import numpy as np
from scipy import linalg as LA
from qiskit.quantum_info.synthesis.two_qubit_decompose import TwoQubitBasisDecomposer
from qiskit.extensions.standard.x import CnotGate
from qiskit.aqua import QuantumInstance,aqua_globals
from qiskit.aqua.operators import WeightedPauliOperator
from qiskit.quantum_info import Pauli
from qiskit.ignis.mitigation.measurement import (complete_meas_cal,tensored_meas_cal,CompleteMeasFitter,TensoredMeasFitter)
import itertools,functools

from math import pi, acos, sqrt
def generate_quantum_instance(run,shots=1024,layout=[0,1],optimization=3,calibration=False):
    if(run=='statevector'):
       # simulazione con algebra lineare e senza barre d'errore
       backend = Aer.get_backend('statevector_simulator')
       quantum_instance = QuantumInstance(backend=backend)
    elif(run=='qasm'):
       # simulazione con algebra lineare e barre d'errore (Monte Carlo sulla distribuzione di probabilita' esatta)
       backend = Aer.get_backend('qasm_simulator')
       quantum_instance = QuantumInstance(backend=backend,shots=shots)
    else:
       # hardware
       IBMQ.load_account()
       provider = IBMQ.get_provider(hub='ibm-q',group='open',project='main')
       backend = provider.get_backend(run)
       if(calibration):
           quantum_instance = QuantumInstance(backend=backend,shots=shots,initial_layout=layout,
                              optimization_level=optimization,measurement_error_mitigation_cls=CompleteMeasFitter)
       else:
           quantum_instance = QuantumInstance(backend=backend,shots=shots,initial_layout=layout,optimization_level=optimization)
    return quantum_instance

def generate_initial_circuit(x0):
    n = len(x0)
    circuit = QuantumCircuit(n,n)
    for i in range(n):
        if(x0[i]==1): circuit.x(i)
    return circuit

def generate_pauli_dictionary():
    pauli_dict = {}
    pauli_dict['I'] = np.zeros((2,2),dtype=complex)
    pauli_dict['X'] = np.zeros((2,2),dtype=complex)
    pauli_dict['Y'] = np.zeros((2,2),dtype=complex)
    pauli_dict['Z'] = np.zeros((2,2),dtype=complex)
    pauli_dict['I'][0,0] = pauli_dict['I'][1,1] = 1.0
    pauli_dict['X'][0,1] = pauli_dict['X'][1,0] = 1.0
    pauli_dict['Z'][0,0] =  1.0
    pauli_dict['Z'][1,1] = -1.0
    pauli_dict['Y'][0,1] = -1j
    pauli_dict['Y'][1,0] =  1j
    # pauli I
    # 1  0
    # 0  1
    # pauli X
    # 0  1
    # 1  0
    # pauli
    # 0 -i
    # i  0
    # pauli Z
    # 1  0
    # 0 -1
    #for k in pauli_dict.keys():
    #    print("Pauli operator ",k)
    #    for r in range(2):
    #        for c in range(2):
    #            print(r,c,pauli_dict[k][r,c])
    #print("-----")
    return pauli_dict

def get_matrices(t,c):
    # 2  0  0  c
    # 0  0  c  0
    # 0  c  0  0
    # c  0  0 -2
    pauli_dict = generate_pauli_dictionary()
    H = np.zeros((4,4))
    H =   np.kron(pauli_dict['Z'],pauli_dict['I']) \
      +   np.kron(pauli_dict['I'],pauli_dict['Z']) \
      + c*np.kron(pauli_dict['X'],pauli_dict['X'])
    U = LA.expm(-1j*t*H)
    #print("Ising matrices ")
    #for r in range(4):
    #    for c in range(4):
    #        print(r,c,H[r,c],U[r,c])
    #print("------")
    return H,U

def exp_itHdis(logfile,circuit,t,c):
        theta = acos(1/c)
        e0 = c-1
        e1= sqrt(c*c+1)
        circuit.z(0)
        circuit.rx(pi/2,0)
        circuit.rx(-pi/2,1)
        circuit.cx(0,1)
        circuit.rx(theta,0)
        circuit.rx(e0-e1,0)
        circuit.ry(e0+e1,0)
        circuit.rx(-theta,0)
        circuit.rz(-theta,1)
        circuit.rz(e0-e1,1)
        circuit.ry(e0+e1,1)
        circuit.rz(theta,1)
        circuit.cx(0,1)
        circuit.rx(-pi/2,0)
        circuit.rx(pi/2,1)
        circuit.z(0)
        logfile.write("time = %f \n" % (t))
        return circuit

def evolve_in_real_time(logfile,qc,t,c):
    # definizione di U3 --- https://qiskit-staging.mybluemix.net/documentation/terra/summary_of_quantum_operations.html
    H,U  = get_matrices(t,c)
    two_qubit_cnot_decompose = TwoQubitBasisDecomposer(CnotGate())
    C = two_qubit_cnot_decompose.__call__(U)
    i,j = 0,1
    parameter_string = []
    for g in C:
        instruction,q1,q2 = g
        if(instruction.name=='u3'):
           t1,t2,t3 = instruction.params
           for x in [t1,t2,t3]: parameter_string.append(round(x,4))
           if(q1[0].index==0): idx = i
           else:               idx = j
           qc.u3(t1,t2,t3,idx)
        if(instruction.name=='cx'):
           if(q1[0].index==0): idx_ctrl = i; idx_targ = j
           else:               idx_ctrl = j; idx_targ = i
           qc.cx(idx_ctrl,idx_targ)
    logfile.write("time = %f \n" % (t))
    #logfile.write("circuit: \n"+str(qc.draw())+"\n")
    ##logfile.write(qc.draw())
    return qc
def Heisen_evolve_bond(logfile,qc,i,j,c,t):
          # https://arxiv.org/pdf/quant-ph/0308006.pdf
          # https://arxiv.org/abs/1909.05701
          i,j=0,1
          qc.cx(i,j);                               qc.barrier()
          qc.rx(-2*c*t-np.pi/2.0,i);                qc.barrier()
          qc.rz(-2*c*t,j);                          qc.barrier()
          qc.h(i);                                  qc.barrier()
          qc.cx(i,j);                               qc.barrier()
          qc.h(i);                                  qc.barrier()
          qc.rz(2*c*t,j);                           qc.barrier()
          qc.cx(i,j);                               qc.barrier()
          qc.rx(np.pi/2.0,i);                       qc.barrier()
          qc.rx(-np.pi/2.0,j);                      qc.barrier()
          return qc

def tomography(logfile,circuit,quantum_instance):
    # misuriamo i 16 operatori di Pauli II XI YI ZI IX XX YX ZX ... ZZ
    # e mettiamo i valori di aspettazione e le barre d'errore su un dizionario dict['ZZ'] = (E[Z],Sqrt[Var[Z]])
    paulis    = [[0,0],[0,1],[1,1],[1,0]]
    names     = ['I','X','Y','Z']
    names_mn  = []
    paulis_mn = []
    #logfile.write("Tomography")
    i,j = 0,1

    provider     = IBMQ.load_account()

    # generiamo gli operatori di Pauli per 2 qubit
    for m,(ax,az) in enumerate(paulis):
        for n,(bx,bz) in enumerate(paulis):
            x_vector = [0]*circuit.n_qubits
            z_vector = [0]*circuit.n_qubits
            x_vector[i] = ax
            z_vector[i] = az
            x_vector[j] = bx
            z_vector[j] = bz
            x_vector    = np.asarray(x_vector,dtype=np.bool)
            z_vector    = np.asarray(z_vector,dtype=np.bool)
            Pmn         = WeightedPauliOperator([(1.0,Pauli(x_vector,z_vector))])
            paulis_mn.append(Pmn)
            names_mn.append(names[m]+names[n])
            #logfile.write("operator %s : " %  (names[m]+names[n]))
            #logfile.write(Pmn.print_details())

    # costruiamo tutti i circuiti per valutare gli operatori di Pauli a 2 qubit
    circuits = []
    for idx,oper in enumerate(paulis_mn):
        c = oper.construct_evaluation_circuit(wave_function=circuit,
            statevector_mode=quantum_instance.is_statevector,
            use_simulator_snapshot_mode=False,
            circuit_name_prefix=str(idx))
        circuits.append(c)
        #provider     = IBMQ.get_provider(hub='ibm-q-internal',group='deployed',project='default')
        #backend      = provider.get_backend('ibmqx2')
        #from qiskit.compiler import transpile
        #transpiled_circuit = transpile(c[0],backend,optimization_level=0)
        #print("circuit for operator ",names_mn[idx])
        #print(transpiled_circuit.draw())

    # calcoliamo i valori di aspettazione degli operatori di Pauli a 2 qubit
    if circuits:
        to_be_simulated_circuits = \
            functools.reduce(lambda x, y: x + y, [c for c in circuits if c is not None])
        result = quantum_instance.execute(to_be_simulated_circuits)

    # versiamoli su un dizionario
    tomography_dict = {}
    for idx,oper in enumerate(paulis_mn):
        mean, std = oper.evaluate_with_result(
                    result=result,statevector_mode=quantum_instance.is_statevector,
                    use_simulator_snapshot_mode=False,
                    circuit_name_prefix=str(idx))
        mean,std = np.real(mean),np.abs(std)
        tomography_dict[names_mn[idx]] = [mean,std]
    for k in tomography_dict.keys():
        logfile.write("Pauli operator %s %s\n"%(k, tomography_dict[k]))

    return tomography_dict

def minus_x_log_x(s):
    if(np.abs(s)<1e-6): return 0.0
    return - s*np.log(s)

def analyze_and_print(logfile,rho,name):
    rho /= rho.shape[0]
    #logfile.write("Density operator "+str(name))
    #for r in range(rho.shape[0]):
    #    for c in range(rho.shape[1]):
    #        rh=rho[r,c]
            #logfile.write("rho[%d,%d]=%f \n" % (r,c,rho[r,c]))
    #logfile.write("Trace %f \n" % np.trace(rho))
    #logfile.write("Hermiticity %f \n" % np.abs(rho-np.conj(rho.T)).max())
    sigma,U = LA.eigh(rho)
    #rho_tilde = np.einsum('ik,k,kl->il',U,sigma,np.conj(U.T))

    #print("spectrum ",sigma)
    # proiezione della rho sull'insieme degli operatori densita'
    sigma   = np.abs(sigma)
    sigma   = [ max(s,0) for s in sigma]
    sigma  /= np.sum(sigma)
    #logfile.write("Eigenvalues \n"+str(sigma))
    P       = np.sum(sigma**2)
    S       = np.sum([minus_x_log_x(s) for s in sigma])
    #logfile.write("Purity, von Neumann entropy %f, %f \n" %(P,S))

    rho_tilde = np.einsum('ik,k,kl->il',U,sigma,np.conj(U.T))

    return P,S,rho_tilde


def process_tomography_dict(logfile,tomography_dict):
    # per un sistema di k qubits,
    # rho = (\sum_m E[Pm] Pm)/2**k
    # dove Pm e' un operatore di Pauli a k qubits
    # ad esempio per 1 qubit
    # rho = (I+a*X+b*Y+c*Z)/2
    # e il vettore (a,b,c) e' il vettore di Bloch

    #for k in tomography_dict.keys():
        #logfile.write("tomography of operator %s" % k)
        #logfile.write(" = %f +/- %f \n" % (tomography_dict[k][0],tomography_dict[k][1]))
    pauli_dict = generate_pauli_dictionary()

    n_sample = 100
    I_vector = np.zeros(n_sample)
    P_A_vector = np.zeros(n_sample)
    P_B_vector = np.zeros(n_sample)
    P_AB_vector = np.zeros(n_sample)
    S_AB_vector = np.zeros(n_sample)
    S_A_vector = np.zeros(n_sample)
    S_B_vector = np.zeros(n_sample)

    for i in range(n_sample):

        rho_AB = np.zeros((4,4),dtype=complex)
        for k in tomography_dict.keys():
            p0,p1   = k[0],k[1]
            ave,std = tomography_dict[k]
            sample  = np.random.normal(ave,std,1)
            rho_AB += sample*np.kron(pauli_dict[p0],pauli_dict[p1])

        P_AB,S_AB,rho_AB = analyze_and_print(logfile,rho_AB,'rho(A,B)')
        # rho(ij,kl) = <ij|rho|kl>
        rho_AB = rho_AB.reshape((2,2,2,2))
        rho_A  = np.einsum('ijkj->ik',rho_AB)
        rho_B  = np.einsum('ijil->jl',rho_AB)
        P_A,S_A,_ = analyze_and_print(logfile,rho_A,'rho(A)')
        P_B,S_B,_ = analyze_and_print(logfile,rho_B,'rho(B)')
        I         = S_A+S_B-S_AB
        I_vector[i] = I
        P_A_vector[i] = P_A
        P_B_vector[i] = P_B
        S_A_vector[i] = S_A
        S_B_vector[i] = S_B
        S_AB_vector[i] = S_AB
        P_AB_vector[i] = P_AB
    Iave,Istd = np.mean(I_vector),np.std(I_vector)
    SABave,SABstd = np.mean(S_AB_vector),np.std(S_AB_vector)
    PABave,PABstd = np.mean(P_AB_vector),np.std(P_AB_vector)
    SAave,SAstd = np.mean(S_A_vector),np.std(S_A_vector)
    SBave,SBstd = np.mean(S_B_vector),np.std(S_B_vector)
    PAave,PAstd = np.mean(P_A_vector),np.std(P_A_vector)
    PBave,PBstd = np.mean(P_B_vector),np.std(P_B_vector)

    logfile.write("I(A,B) = %f +/- %f \n " % (Iave,Istd))
    logfile.write("S(A,B) = %f +/- %f \n " % (SABave,SABstd))
    logfile.write("P(A,B) = %f +/- %f \n " % (PABave,PABstd))
    logfile.write("S(A) = %f +/- %f \n " % (SAave,SAstd))
    logfile.write("S(B) = %f +/- %f \n " % (SBave,SBstd))
    logfile.write("P(A) = %f +/- %f \n " % (PAave,PAstd))
    logfile.write("P(B) = %f +/- %f \n " % (PBave,PAstd))
    logfile.write("rho(AB) = %s \n " % rho_AB)
    
    return Iave, Istd
