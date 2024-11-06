# https://github.com/heineaabo/QuantumAlgorithms/blob/9ef038b5b2b721efada8360e76382b055a1fd38b/quantum_algorithms/systems/PairingModel/thesis_work/ibm/z_only.py
import numpy as np

import sys
sys.path.append('../..')
from matrix import get_pairing_matrix
sys.path.append('../../../..')
sys.path.append('../../../../optimizers')
sys.path.append('../../../../../../QuantumCircuitOptimizer')
from quantum_circuit import QuantumCircuit,SecondQuantizedHamiltonian,PairingHamiltonian,CircuitList,PauliString,X
from fci import FCI

from vqe import VQE
from optimizers import Minimizer

# IBM PREP
import qiskit as qk
qk.IBMQ.load_account()
provider = qk.IBMQ.get_provider('ibm-q')
#qcomp = provider.get_backend('ibmq_essex')
qcomp = provider.get_backend('ibmq_london')



l = 4     # number of spin orbitals / number of qubits
n = 2     # Number of occupied spin orbitals
delta = 1 # Level spacing
g = 1     # Interaction strength

h,v = get_pairing_matrix(n,l,delta,g)

Efci = FCI(n,l,h,v)
print('FCI energy :',Efci)

import time
t1 = time.time()
pairing =  PairingHamiltonian(n,l,h,v)
t2 = time.time()
print('Time:',t2-t1)

pairing.group_paulis(qwc=True,gc=True)
#pairing._circuit_list = CircuitList()
#pairing._circuit_list.append(PauliString(-0.1,[X(),X(),X(),X()],[0,1,2,3]))

#theta = [5.829889373194686] # Hardcode good parameter
theta = [5.91985576]


#options = {'count_states':False,'shots':1000,'print':True}
options_i = {'shots':2000,'optimization_level':1,
           #'noise_model':True,
           #'meas_fit':True,
           # 'device':'ibmq_london',
           'print':True}
options_q = {'shots':1000,
           #'seed':1,
           'optimization_level':1,
           'backend':qcomp,
           'ibmq':True,
           #'device':'ibmq_london',
           #'meas_fit':True,
           #'layout':[0,3,2,1],  
           'layout':[0,1,2,3],  
           #'layout':[1,0,2,3],  
           #'noise_model':True,
           'basis_gates':True,
           #'coupling_map':True,
           'print':True}
#model = VQE(pairing,Minimizer('Cobyla',tol=1e-05),'UCCD',options=options_i)
#modelq = VQE(pairing,Minimizer('Cobyla',tol=1e-05),'UCCD',options=options_q)
#model = VQE(pairing,Minimizer('Cobyla',tol=1e-05),'RYPAIRING',options=options_i)
modelq = VQE(pairing,Minimizer('Cobyla',tol=1e-05,max_iter=70),'RYPAIRING',options=options_q)
theta = modelq.optimize()
print(theta)
#print(model.get_mean(theta))
#print('simulation')
#print(model.expval(theta))
#print('Real quantum')
#print(modelq.expval(theta))

