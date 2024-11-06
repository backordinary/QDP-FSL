# https://github.com/Hirmay/Simulation-of-LiH-molecule-s-Energy/blob/5a680c4c018ca494ba7e7af38f775044bcf42f02/Qiskit%20Summer%20School%20Final%20Project%20VQE.py
#!/usr/bin/env python
# coding: utf-8

# In[1]:


from qiskit import BasicAer, Aer, IBMQ
from qiskit.aqua import QuantumInstance, aqua_globals
from qiskit.aqua.algorithms import VQE, ExactEigensolver
from qiskit.aqua.components.initial_states import Zero
from qiskit.aqua.components.optimizers import COBYLA, L_BFGS_B, SLSQP, SPSA
from qiskit.aqua.components.variational_forms import RY, RYRZ, SwapRZ
from qiskit.aqua.operators import WeightedPauliOperator, Z2Symmetries
from qiskit.chemistry import FermionicOperator
from qiskit.chemistry.drivers import PySCFDriver, UnitsType
from qiskit.chemistry.components.variational_forms import UCCSD
from qiskit.chemistry.components.initial_states import HartreeFock

from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import QuantumError, ReadoutError
from qiskit.providers.aer.noise.errors import pauli_error
from qiskit.providers.aer.noise.errors import depolarizing_error
from qiskit.providers.aer.noise.errors import thermal_relaxation_error

from qiskit.providers.aer import noise
provider = IBMQ.load_account()

import numpy as np
import matplotlib.pyplot as plt
from functools import partial


# # Qiskit Summer School Final Project: VQE
# 
# #### For this optional final challenge, you will be designing your own implementation of a variational quantum eigensolver (VQE) algorithm that simulates the ground state energy of the Lithium Hydride (LiH) molecule. Through out this challenge, you will be able to make choices on how you want to compose your simulation and what is the final deliverable that you want to showcase to your classmates and friends.

# # Defining your molecule:
# In this challenge we will focus on LiH using the sto3g basis with the PySCF driver, which can be described in Qiskit as follows, where 'inter_dist' is the interatomic distance.

# In[2]:


driver = PySCFDriver(atom='Li .0 .0 .0; H .0 .0' + str(inter_dist), unit=UnitsType.ANGSTROM, charge=0, spin=0, basis='sto3g')


# We also setup the molecular orbitals to be considered and can reduce the problem size when we map to the qubit Hamiltonian so the amount of time required for the simulations are reasonable for a laptop computer.

# In[3]:


# please be aware that the idx here with respective to original idx
freeze_list = [0]
remove_list = [-3, -2] # negative number denotes the reverse order


# #### Once you have computed the qubit operations for LiH, you can use the following function to classical solve for the exact solution. This is used just to compare how well your VQE approximation is performing.

# In[4]:


#Classically solve for the lowest eigenvalue
def exact_solver(qubitOp):
    ee = ExactEigensolver(qubitOp)
    result = ee.run()
    ref = result['energy']
    print('Reference value: {}'.format(ref))
    return ref


# Here we ask you to use the `statevector_simulator` as the simulation backend for your VQE algorithm.

# In[13]:


backend = BasicAer.get_backend('statevector_simulator')


# ### Now you can start choosing the components that make up your VQE algorithm!
# 
# #### 1. Optimizers
# The most commonly used optimizers are `COBYLA`, `L_BFGS_B`, `SLSQP` and `SPSA`. 
# 
# #### 2. Qubit mapping
# There are several different mappings for your qubit Hamiltonian, `parity`, `bravyi_kitaev`, `jordan_wigner`, which in some cases can allow you to further reduce the problem size.
# 
# #### 3. Initial state
# There are different initial state that you can choose to start your simulation. Typically people choose from the zero state 
# `init_state = Zero(qubitOp.num_qubits)` 
# and the UCCSD initial state
# `HartreeFock(qubitOp.num_qubits, num_spin_orbitals, num_particles, map_type, qubit_reduction)`
# 
# #### 4. Parameterized circuit
# There are different choices you can make on the form of variational forms of your parameterized circuit.
# 
# `UCCSD_var_form = UCCSD(num_qubits, depth=depth, num_orbitals=num_spin_orbitals, num_particles=num_particles)`
#     
# `RY_var_form = RY(num_qubits, depth=depth)`
#     
# `RYRZ_var_form = RYRZ(num_qubits, depth=depth)`
#     
# `swaprz_var_form = SwapRZ(num_qubits, depth=depth)`
# 
# #### 5. Simulation backend
# There are different simulation backends that you can use to perform your simulation
# 
# `backend = BasicAer.get_backend('statevector_simulator')`
# 
# `backend=Aer.get_backend('qasm_simulator')`

# ### Compare the convergence of different choices for building your VQE algorithm
# 
# Among the above choices, which combination do you think would out perform others and give you the lowest estimation of LiH ground state energy with the quickest convergence? Compare the results of different combinations against each other and against the classically computed exact solution at a fixed interatomic distance, for example `inter_dist=1.6`. 
# 
# To access the intermediate data during the optimization, you would need to utilize the `callback` option in the VQE function:
# 
# `def store_intermediate_result(eval_count, parameters, mean, std):
#             counts.append(eval_count)
#             values.append(mean)
#             params.append(parameters)
#             deviation.append(std)`
#             
# `algo = VQE(qubitOp, var_form, optimizer, callback=store_intermediate_result)`
# 
# `algo_result = algo.run(quantum_instance)`
# 
# An example of comparing the performance of different optimizers while using the RY variational ansatz could like the following:
# ![RY_error.png](attachment:RY_error.png)
# ![RY_convergence.png](attachment:RY_convergence.png)

# ### Compute the ground state energy of LiH at various different interatomic distances
# By changing the parameter `inter_dist`, you can use your VQE algorithm to calculate the ground state energy of LiH at various interatomic distances, and potentially produce a plot as you are seeing here. Note that the VQE results are very close to the exact results, and so the exact energy curve is hidden by the VQE curve.
# <img src="attachment:VQE_dist.png" width="600">

# ### How does your VQE algorithm perform in the presence of noise?
# Trying importing the noise model and qubit coupling map of a real IBM quantum device into your simulation. You can use the imported noise model in your simulation by passing it into your quantum instance. You can also try enabling error mitigation in order to lower the effect of noise on your simulation results.

# In[5]:


#Define our noise model based on the ibmq_essex chip
chip_name = 'ibmq_essex'
device = provider.get_backend(chip_name)
coupling_map = device.configuration().coupling_map
noise_model = noise.device.basic_device_noise_model(device.properties())
basis_gates = noise_model.basis_gates


# An example of comparing the energy convergence of using SPSA and COBYLA with the ibmq_essex noise model could look like the following
# ![noise.png](attachment:noise.png)

# ### Now given the choices you have made above, try writing your own VQE algorithm in Qiskit. You can find an example of using Qiskit to simuate molecules with VQE [here](https://qiskit.org/textbook/ch-applications/vqe-molecules.html).

# In[6]:


def exact_solver(qubitOp):
    ee = ExactEigensolver(qubitOp)
    result = ee.run()
    ref = result['energy']
    #print('Reference value: {}'.format(ref))
    return ref

# Define your function for computing the qubit operations of LiH
def compute_LiH_qubitOp(map_type, inter_dist=1.6, basis='sto3g'):
    
    # Specify details of our molecule
    driver = PySCFDriver(atom='Li .0 .0 .0; H .0 .0 ' + str(inter_dist), unit=UnitsType.ANGSTROM, charge=0, spin=0, basis=basis)

    # Compute relevant 1 and 2 body integrals.
    molecule = driver.run()
    h1 = molecule.one_body_integrals
    h2 = molecule.two_body_integrals
    nuclear_repulsion_energy = molecule.nuclear_repulsion_energy
    
    num_particles = molecule.num_alpha + molecule.num_beta
    num_spin_orbitals = molecule.num_orbitals * 2
    #print("HF energy: {}".format(molecule.hf_energy - molecule.nuclear_repulsion_energy))
    #print("# of electrons: {}".format(num_particles))
    #print("# of spin orbitals: {}".format(num_spin_orbitals))

    # Please be aware that the idx here with respective to original idx
    freeze_list = [0]
    remove_list = [-3, -2] # negative number denotes the reverse order
    
    # Prepare full idx of freeze_list and remove_list
    # Convert all negative idx to positive
    remove_list = [x % molecule.num_orbitals for x in remove_list]
    freeze_list = [x % molecule.num_orbitals for x in freeze_list]
    
    # Update the idx in remove_list of the idx after frozen, since the idx of orbitals are changed after freezing
    remove_list = [x - len(freeze_list) for x in remove_list]
    remove_list += [x + molecule.num_orbitals - len(freeze_list)  for x in remove_list]
    freeze_list += [x + molecule.num_orbitals for x in freeze_list]

    # Prepare fermionic hamiltonian with orbital freezing and eliminating, and then map to qubit hamiltonian
    # and if PARITY mapping is selected, reduction qubits
    energy_shift = 0.0
    qubit_reduction = True if map_type == 'parity' else False

    ferOp = FermionicOperator(h1=h1, h2=h2)
    if len(freeze_list) > 0:
        ferOp, energy_shift = ferOp.fermion_mode_freezing(freeze_list)
        num_spin_orbitals -= len(freeze_list)
        num_particles -= len(freeze_list)
    if len(remove_list) > 0:
        ferOp = ferOp.fermion_mode_elimination(remove_list)
        num_spin_orbitals -= len(remove_list)

    qubitOp = ferOp.mapping(map_type=map_type)
    qubitOp = Z2Symmetries.two_qubit_reduction(qubitOp, num_particles) if qubit_reduction else qubitOp
    qubitOp.chop(10**-10)
    shift=energy_shift+nuclear_repulsion_energy
    return qubitOp, num_spin_orbitals, num_particles, qubit_reduction,shift
#Backend
backend = BasicAer.get_backend("statevector_simulator")
#Maping Type
map_type='parity'

#distance range
distance = np.arange(0.5, 4.25, 0.25)
#List of values
exact_energies = []
vqe_energies = []
acc1=[]
for dist in distance:
 qubitOp, num_spin_orbitals, num_particles, qubit_reduction,shift = compute_LiH_qubitOp(map_type,inter_dist=dist)
# Classically solve for the exact solution and use that as your reference value
 ref = exact_solver(qubitOp)
 exact_energies.append(np.real(ref)+shift) 
# Specify your initial state
 initial_state = HartreeFock(
        num_spin_orbitals,
        num_particles,
        qubit_mapping='bravyi_kitaev'
    ) 

# Select a state preparation ansatz
# Equivalently, choose a parameterization for our trial wave function.
 var_form =  UCCSD(
        num_orbitals=num_spin_orbitals,
        num_particles=num_particles,
        initial_state=initial_state,
        qubit_mapping='parity'
    )

# Choose where to run/simulate our circuit
 quantum_instance = QuantumInstance(backend=backend, 
                                   shots=1000)
# Choose the classical optimizer
 optimizer = COBYLA(maxiter=200, tol=0.0001)

# Run your VQE instance
 vqe = VQE(qubitOp, var_form, optimizer)

# Now compare the results of different compositions of your VQE algorithm!
 vqe_result = np.real(vqe.run(backend)['eigenvalue'])
 vqe_energies.append(vqe_result+shift)
 print("Interatomic Distance:", np.round(dist, 2), "VQE Result: ", vqe_result,", Exact Result: ",ref,", Accuracy: ",100-(abs(abs(ref-vqe_result)/ref)*100))
 acc1.append(100-(abs(abs(ref-vqe_result)/ref)*100))


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
# Importing standard Qiskit libraries and configuring account
from qiskit import QuantumCircuit, execute, Aer, IBMQ
from qiskit.compiler import transpile, assemble
from qiskit.tools.jupyter import *
from qiskit.visualization import *
# Loading your IBM Q account(s)
provider = IBMQ.load_account()


# # Noise-Less Graph 

# In[17]:


plt.plot(distance, exact_energies, label="Exact Energy")
plt.plot(distance, vqe_energies, label="VQE Energy")
plt.xlabel('Atomic distance (Angstrom)')
plt.ylabel('Energy')
plt.legend()
plt.show()


# In[18]:


plt.plot(distance, acc1, label="Accuracy(NoiseLess)")
plt.xlabel('Atomic distance (Angstrom)')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# # Simulation with noise

# In[21]:


def exact_solver(qubitOp):
    ee = ExactEigensolver(qubitOp)
    result = ee.run()
    ref = result['energy']
    #print('Reference value: {}'.format(ref))
    return ref

# Define your function for computing the qubit operations of LiH
def compute_LiH_qubitOp(map_type, inter_dist=1.6, basis='sto3g'):
    
    # Specify details of our molecule
    driver = PySCFDriver(atom='Li .0 .0 .0; H .0 .0 ' + str(inter_dist), unit=UnitsType.ANGSTROM, charge=0, spin=0, basis=basis)

    # Compute relevant 1 and 2 body integrals.
    molecule = driver.run()
    h1 = molecule.one_body_integrals
    h2 = molecule.two_body_integrals
    nuclear_repulsion_energy = molecule.nuclear_repulsion_energy
    
    num_particles = molecule.num_alpha + molecule.num_beta
    num_spin_orbitals = molecule.num_orbitals * 2
    #print("HF energy: {}".format(molecule.hf_energy - molecule.nuclear_repulsion_energy))
    #print("# of electrons: {}".format(num_particles))
    #print("# of spin orbitals: {}".format(num_spin_orbitals))

    # Please be aware that the idx here with respective to original idx
    freeze_list = [0]
    remove_list = [-3, -2] # negative number denotes the reverse order
    
    # Prepare full idx of freeze_list and remove_list
    # Convert all negative idx to positive
    remove_list = [x % molecule.num_orbitals for x in remove_list]
    freeze_list = [x % molecule.num_orbitals for x in freeze_list]
    
    # Update the idx in remove_list of the idx after frozen, since the idx of orbitals are changed after freezing
    remove_list = [x - len(freeze_list) for x in remove_list]
    remove_list += [x + molecule.num_orbitals - len(freeze_list)  for x in remove_list]
    freeze_list += [x + molecule.num_orbitals for x in freeze_list]

    # Prepare fermionic hamiltonian with orbital freezing and eliminating, and then map to qubit hamiltonian
    # and if PARITY mapping is selected, reduction qubits
    energy_shift = 0.0
    qubit_reduction = True if map_type == 'parity' else False

    ferOp = FermionicOperator(h1=h1, h2=h2)
    if len(freeze_list) > 0:
        ferOp, energy_shift = ferOp.fermion_mode_freezing(freeze_list)
        num_spin_orbitals -= len(freeze_list)
        num_particles -= len(freeze_list)
    if len(remove_list) > 0:
        ferOp = ferOp.fermion_mode_elimination(remove_list)
        num_spin_orbitals -= len(remove_list)

    qubitOp = ferOp.mapping(map_type=map_type)
    qubitOp = Z2Symmetries.two_qubit_reduction(qubitOp, num_particles) if qubit_reduction else qubitOp
    qubitOp.chop(10**-10)
    shift=energy_shift+nuclear_repulsion_energy
    return qubitOp, num_spin_orbitals, num_particles, qubit_reduction,shift
#Backend
backend = BasicAer.get_backend("qasm_simulator")
#Maping Type
map_type='parity'

#distance range
distance = np.arange(0.5, 4.25, 0.25)
#List of values
exact_energies = []
vqe_energies = []
acc1=[]
for dist in distance:
 qubitOp, num_spin_orbitals, num_particles, qubit_reduction,shift = compute_LiH_qubitOp(map_type,inter_dist=dist)
# Classically solve for the exact solution and use that as your reference value
 ref = exact_solver(qubitOp)
 exact_energies.append(np.real(ref)+shift) 
# Specify your initial state
 initial_state = HartreeFock(
        num_spin_orbitals,
        num_particles,
        qubit_mapping='bravyi_kitaev'
    ) 

# Select a state preparation ansatz
# Equivalently, choose a parameterization for our trial wave function.
 var_form =  UCCSD(
        num_orbitals=num_spin_orbitals,
        num_particles=num_particles,
        initial_state=initial_state,
        qubit_mapping='parity'
    )

# Choose where to run/simulate our circuit
 quantum_instance = QuantumInstance(backend=backend, 
                                   shots=1000)
# Choose the classical optimizer
 optimizer = COBYLA(maxiter=200, tol=0.0001)

# Run your VQE instance
 vqe = VQE(qubitOp, var_form, optimizer)

# Now compare the results of different compositions of your VQE algorithm!
 vqe_result = np.real(vqe.run(backend)['eigenvalue'])
 vqe_energies.append(vqe_result+shift)
 print("Interatomic Distance:", np.round(dist, 2), "VQE Result: ", vqe_result,", Exact Result: ",ref,", Accuracy: ",100-(abs(abs(ref-vqe_result)/ref)*100))
 acc1.append(100-(abs(abs(ref-vqe_result)/ref)*100))


# In[22]:


plt.plot(distance, exact_energies, label="Exact Energy")
plt.plot(distance, vqe_energies, label="VQE Energy")
plt.xlabel('Atomic distance (Angstrom)')
plt.ylabel('Energy')
plt.legend()
plt.show()


# In[23]:


plt.plot(distance, acc1, label="Accuracy(NoiseLess)")
plt.plot(distance, acc2, label="Accuracy(Noisy)")
plt.xlabel('Atomic distance (Angstrom)')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[16]:


import qiskit.tools.jupyter
get_ipython().run_line_magic('qiskit_version_table', '')
get_ipython().run_line_magic('qiskit_copyright', '')


# In[ ]:




