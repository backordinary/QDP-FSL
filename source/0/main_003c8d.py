# https://github.com/vinitX/RBM/blob/8f43268496a5ad0ca8adabe4e793627e9556a96c/main.py
from gate_quantum import *
from gradient import *
import sys
import csv
import qiskit
import numpy as np
from qiskit import IBMQ
from qiskit import *
from qiskit.quantum_info import Statevector
from collections import Counter
import scipy.sparse
from qiskit import IBMQ, assemble, transpile
from qiskit.ignis.mitigation.measurement import (complete_meas_cal,CompleteMeasFitter)

n=int(sys.argv[1])
m=int(sys.argv[2])
Ham_file= sys.argv[3]
k_index = int(sys.argv[4]) 
it_max =int(sys.argv[5])

transfer_learning=sys.argv[6] if len(sys.argv) >= 7 else False
transf_indx =sys.argv[7] if len(sys.argv) >= 8 else None


np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
matrixglobal = np.load(Ham_file)
np.load = np_load_old

epsilon=1*pow(10,-4)

#n = 1 # visible
#m = 2 # hidden

def rbm_new(k_index,it_max):  
    flag=0
    rate = 0.005
#    c2gate1,c2gate2,c2gate3,c2gate4 = c2gate(n,m)
#    np.save('gates1.npy',(c2gate1,c2gate2,c2gate3,c2gate4))

    # Hamiltonian gets extracted and min eigenvalue printed
    #*******************************************************
    matrix1=(matrixglobal)
    #print('matrix1 = ', np.shape(matrix1))
    eigval_act, eigvec_act= np.linalg.eigh(matrix1)
    
    eigval_act, eigvec_act = eigval_act[np.argsort(eigval_act)], eigvec_act[:, np.argsort(eigval_act)]
    Hamiltonian = matrix1[::-1][::-1]
    
    # Parameters/state gets initialized or transfer learning initiated
    #**********************************************************
    
    if transfer_learning !=False: 
        para1 = transfer_para(n,m,rate, transf_indx)
    else:
        para1 = initial_para(n,m,rate)

    state = initial_state(n,m)
    success = 1
    np.save('parameters'+str(k_index)+'_exc_ini.npy', para1)
    
    result_filename = "result"+str(k_index)+"_exc.csv"
    property_filename = "Final_prop"+str(k_index)+"_exc.csv"
    file_header_allit = ['it', 'RBM val  ', 'Act val  ', 'Error  '] 
    

    # Looping for iterations starts
    #******************************
    with open(result_filename, "w") as csv_it:
        writer=csv.DictWriter(csv_it, delimiter='\t', fieldnames=file_header_allit)
        writer.writeheader()
        for iteration in range(it_max):
            #rate = 0.01
            state_p, k = calculate_gate(para1,n,m)    # does Gibbs sampling 
            state_sign = sign_prog(state,para1)     # computes tanh sign function 
            update_value=back(state_p,state_sign,para1,state,Hamiltonian)    # updates required for parameters for stochastic gradient
            para1=update(para1,update_value,rate)  # updates all parameter values and new parametrs formed for next cycle
        
            np.save('parameters'+str(k_index)+'_exc.npy', para1)

            if iteration%1==0:
               phi = np.sqrt(np.sum(state_p,axis=1,keepdims=True))*state_sign          #state vector

               prob = phi * np.conjugate(phi)      #probability

               E_loc=np.sum(Hamiltonian*phi.T,axis=1,keepdims=True)/((phi==0) + phi) * (phi!=0)    # computes H|\phi>

               RBM_energy = np.real(np.sum(E_loc*prob)/np.sum(prob))   # energy
            
               Error = np.abs(RBM_energy-eigval_act[0])
               
               writer.writerow({file_header_allit[0]:iteration, file_header_allit[1]:RBM_energy, file_header_allit[2]:eigval_act[0], file_header_allit[3]:Error})

               #results printing
               if iteration == it_max-1:
                     with open(property_filename, "a") as prop:
                          if (Error <= epsilon):
                               prop.write("Calculations converged with Error={}. No Trans Learning req".format(Error))
                          else:
                               prop.write("Calculations NOT converged with Error={}. Trans Learning req".format(Error))
                               break
                          prop.write("\n")
                          prop.write("\n")
                          prop.write("Final RBM Energy ={}".format(RBM_energy))
                          prop.write("\n")
                          prop.write("\n")
                          prop.write("Actual Ham Energy ={}".format(eigval_act[0]))
                          prop.write("\n")
                          prop.write("\n")
                          prop.write("Final RBM state(Amp, phase) =")
                          prop.write("\n")
                          prop.write("\n")
                          for elem in phi:
                              amp=np.absolute(elem)/np.sqrt(np.sum([k*np.conj(k) for k in phi]))
                              phase=np.angle(elem, deg=True)
                              prop.write(str(np.real(amp[0]))+"\t"+str(phase[0]))
                              prop.write("\n")
                              prop.write("\n")
                          
                          prop.write("Actual Ham state(Amp, phase) =")
                          prop.write("\n")
                          prop.write("\n")
                          for elem in eigvec_act[:,0]:
                              amp=np.absolute(elem)/np.sqrt(np.sum([k*np.conj(k) for k in eigvec_act[:,0]]))
                              phase=np.angle(elem, deg=True)
                              prop.write(str(np.real(amp))+"\t"+str(phase))
                              prop.write("\n")
                              prop.write("\n")
                          
                         # fidelity=np.trace(np.dot(np.outer(eigvec_act[:,0],eigvec_act[:,0]), np.outer(phi[:],phi[:]))) 
                          fidelity=np.sqrt(np.abs(np.dot(np.conj(eigvec_act[:,0])/np.sqrt(np.sum([k*np.conj(k) for k in eigvec_act[:,0]])), phi/np.sqrt(np.sum([k*np.conj(k) for k in phi])))))
                          prop.write("Fidelity =")
                          prop.write("\n")
                          prop.write("\n")
                          prop.write(str(fidelity[0])) 
    return None


rbm_new(k_index,it_max)

