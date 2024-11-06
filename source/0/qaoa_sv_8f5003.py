# https://github.com/huckstar/QAOA/blob/304244a5afe59713673bcc82e67dc3faed5aac93/qaoa_sv.py
from qiskit import QuantumCircuit, execute, Aer, IBMQ , ClassicalRegister,QuantumRegister,execute
from qiskit.compiler import transpile, assemble
#from qiskit.tools.jupyter import *
from qiskit.visualization import *
#from ibm_quantum_widgets import execute
import math
import time
import matplotlib.pyplot as plt
import numpy as np
import qiskit
import networkx as nx
from fractions import Fraction
from common import *
from scipy.optimize import minimize
from collections import defaultdict
import matplotlib.pyplot as plt

n = 10   #set paramters of max -k-vertex
k = 5

num_graph = 2
mixer = 1 #1:complete; 2: ring; 3: Grover
max_layer = 2

backend = Aer.get_backend('statevector_simulator')
################
average_ratio1=[]
average_ratio2=[]
average_ratio3=[]
average_depth1=[]
average_depth2=[]
average_depth3=[]
for layer in range(max_layer):
    ratio={'1':[],'2':[],'3':[]}
    
    depth={'1':[],'2':[],'3':[]}
    
    for idex in range(num_graph):

        G = nx.gnp_random_graph(n, 0.5)
        #get the circuit
        #qc.draw(output='mpl',filename='my_circuit2.png')
        p = layer+1
        for mixer in (1,2,3):

            obj = get_black_box_objective_sv(G,p,n,k,mixer) #optimized target
        


            init_point = np.zeros((2*p,), dtype=int)
            res_sample = minimize(obj,init_point,method='COBYLA',options={'maxiter':2500,'disp':True})
        

            optimal_theta = res_sample['x']
            qc = get_qaoa_circuit_sv(G,optimal_theta[:p],optimal_theta[p:],n,k,mixer)
        
            #counts = invert_counts(execute(qc,backend).result().get_counts())
        


            
        
            sv = get_adjusted_state(execute(qc, backend).result().get_statevector())
            Fp = compute_maxkvertex_energy_sv(sv,G)
            maxvalue = max(-maxkvertex_obj(np.array([x for x in l]), G) for l,v in state_to_ampl_counts(sv).items())
            #plt.bar(x,y)
            #plt.show()
            #plt.savefig('test1.png')
            ratio[str(mixer)].append(-Fp/maxvalue)
            depth[str(mixer)].append(qc.depth())
    average_ratio1.append(np.mean(ratio['1']))
    average_depth1.append(np.mean(depth['1']))
    average_ratio2.append(np.mean(ratio['2']))
    average_depth2.append(np.mean(depth['2']))
    average_ratio3.append(np.mean(ratio['3']))
    average_depth3.append(np.mean(depth['3']))


fig, axs = plt.subplots(2)
axs[0].plot(range(max_layer),average_ratio1, label = 'complete')
axs[0].plot(range(max_layer),average_ratio2, label = 'ring')
axs[0].plot(range(max_layer),average_ratio3, label = 'Grover')
axs[0].set(xlabel = 'layer',ylabel = 'ratio')
axs[0].legend()
axs[1].plot(average_depth1,average_ratio1, label = 'complete')
axs[1].plot(average_depth2,average_ratio2, label = 'ring')
axs[1].plot(average_depth3,average_ratio3, label = 'Grover')
axs[1].set(xlabel = 'depth',ylabel = 'ratio')
axs[1].legend()
plt.show()
plt.savefig('test_num_10_4.png')
