# https://github.com/jasonlarkin/mol2qpu/blob/359b525698958adfbbb8965dc4512ae0fecdb2d5/H2_FCI.py
#!/usr/bin/env python
# coding: utf-8

# In[29]:


import argparse
import numpy as np
import pickle
import types
from qiskit import Aer, BasicAer
from qiskit.aqua import aqua_globals, QuantumInstance
from qiskit.aqua.algorithms import ExactEigensolver, VQE
from qiskit.aqua.components.optimizers import SPSA, COBYLA, L_BFGS_B
from qiskit.aqua.components.variational_forms import RY, RYRZ
#from qiskit.aqua.operators import Z2Symmetries
from qiskit.chemistry import FermionicOperator
#from qiskit.chemistry.core import Hamiltonian, QubitMappingType, TransformationType
from qiskit.chemistry.drivers import PySCFDriver, UnitsType

args = types.SimpleNamespace()
#args.algorithm='VQE' 
args.basis_set='ccpvdz' 
#args.basis_set='sto3g' 
args.max_parallel_threads=10 
args.molecule='H2' 
args.num_shots=1000
args.outpath_mol='/home/ubuntu/mol2qpu/output/operators/molecule/'
args.outpath_ferm='/home/ubuntu/mol2qpu/output/operators/fermionic/'
args.outpath_qub='/home/ubuntu/mol2qpu/output/operators/qubit/'
args.outpath_vqe='/home/ubuntu/mol2qpu/output/VQE_results/'
args.qubitmapping_type='bravyi_kitaev'
#args.qubitmapping_type='jordan_wigner'
#args.qubitmapping_type='parity'
#args.random_seed=750 
#args.two_qubit_reduce = True
#args.vqe_aer = True
args.vqe_depth=1 
args.vqe_entangler='linear' 
#args.vqe_max_iter=2
#args.vqe_opt_params = False
#args.vqe_optimizer='SPSA' 
#args.vqe_sim = True
#args.vqe_var_form='RY' 


# In[30]:


## load qmolecule
d_string = '0pt55'
filename_mol = args.outpath_mol + args.molecule + '_' + args.basis_set + '_' + d_string + '_MOLE' + '.pkl'
qmolecule = pickle.load(open(filename_mol,'rb'))

## load operators
filename_ferm = args.outpath_ferm + args.molecule + '_' + args.basis_set + '_' + d_string + '_FERM' + '.pkl'
filename_qub = args.outpath_qub + args.molecule + '_' + args.basis_set + '_' + d_string + '_QUBIT_bk' + '.pkl'
ferOp = pickle.load(open(filename_ferm, 'rb'))
qubitOp = pickle.load(open(filename_qub, 'rb'))


# In[31]:


print("molecule file = ", filename_mol)
print("qubit file = ", filename_qub)
num_particles = qmolecule.num_alpha + qmolecule.num_beta
print("num orbitals = ", qmolecule.num_orbitals)
print("num_particles = ", num_particles)
print("num qubits = ", qubitOp.num_qubits)


# In[41]:


backend = Aer.get_backend('qasm_simulator')
quantum_instance = QuantumInstance(circuit_caching=True, 
                                   backend=backend,
                                   backend_options={'max_parallel_threads': args.max_parallel_threads,                                                                             'max_parallel_experiments': 0, 
                                                    'shots': args.num_shots})
optimizer = SPSA(max_trials=200)
var_form = RY(qubitOp.num_qubits, depth=args.vqe_depth, entanglement=args.vqe_entangler) 
algo = VQE(qubitOp, var_form, optimizer)
result = algo.run(quantum_instance)  
## save VQE results
filename_vqe = args.outpath_vqe              + args.molecule + '_'              + args.basis_set + '_'              + d_string + '_VQE_SPSA_RY_lin_bk_'              + str(args.num_shots) + '.pkl'
filehandler_vqe = open(filename_vqe, 'wb')
pickle.dump(result, filehandler_vqe)
print("saved file = ", filename_vqe)


# In[32]:


print("VQE settings = ", algo.print_settings())
print("VQE result = ", result)


# In[28]:


## load VQE results
#print(filename_vqe)
#result_load = pickle.load(open(filename_vqe,'rb'))
#result_load

