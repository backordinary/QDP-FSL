# https://github.com/SamirFarhat17/quantum-computer-programming-ibm/blob/eeb446026f480cdb48e4dc9c6d23b825300493c9/nature-experiments/sampling_pes.py
# import common packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial

# qiskit
from qiskit.aqua import QuantumInstance
from qiskit import BasicAer
from qiskit.aqua.algorithms import NumPyMinimumEigensolver, VQE, IQPE
from qiskit.aqua.components.optimizers import SLSQP
from qiskit.circuit.library import ExcitationPreserving
from qiskit import BasicAer
from qiskit.aqua.algorithms import NumPyMinimumEigensolver, VQE
from qiskit.aqua.components.optimizers import SLSQP

# chemistry components
from qiskit.chemistry.components.initial_states import HartreeFock
from qiskit.chemistry.components.variational_forms import UCCSD
from qiskit.chemistry.drivers import PySCFDriver, UnitsType
from qiskit.chemistry.core import Hamiltonian, TransformationType, QubitMappingType
from qiskit.aqua.algorithms import VQAlgorithm, VQE, MinimumEigensolver
from qiskit.chemistry.transformations import FermionicTransformation
from qiskit.chemistry.drivers import PySCFDriver
from qiskit.chemistry.algorithms.ground_state_solvers import GroundStateEigensolver
from qiskit.chemistry.algorithms.pes_samplers.bopes_sampler import BOPESSampler
from qiskit.chemistry.drivers.molecule import Molecule
from qiskit.chemistry.algorithms.pes_samplers.extrapolator import *

import warnings
warnings.simplefilter('ignore', np.RankWarning)

ft = FermionicTransformation()
driver = PySCFDriver()
solver = VQE(quantum_instance=
             QuantumInstance(backend=BasicAer.get_backend('statevector_simulator')))
me_gsc = GroundStateEigensolver(ft, solver)
stretch1 = partial(Molecule.absolute_stretching, atom_pair=(1, 0))
mol = Molecule(geometry=[('H', [0., 0., 0.]),
                        ('H', [0., 0., 0.3])],
                       degrees_of_freedom=[stretch1],
                       )

# pass molecule to PSYCF driver
driver = PySCFDriver(molecule=mol)
print(mol.geometry)

mol.perturbations = [0.2]
print(mol.geometry)

mol.perturbations = [0.6]
print(mol.geometry)

distance1 = partial(Molecule.absolute_distance, atom_pair=(1, 0))
mol = Molecule(geometry=[('H', [0., 0., 0.]),
                        ('H', [0., 0., 0.3])],
                       degrees_of_freedom=[distance1],
                       )

# pass molecule to PSYCF driver
driver = PySCFDriver(molecule=mol)

# Specify degree of freedom (points of interest)
points = np.linspace(0.25, 2, 30)
results_full = {} # full dictionary of results for each condition
results = {} # dictionary of (point,energy) results for each condition
conditions = {False: 'no bootstrapping', True: 'bootstrapping'}


for value, bootstrap in conditions.items():
    # define instance to sampler
    bs = BOPESSampler(
        gss=me_gsc
        ,bootstrap=value
        ,num_bootstrap=None
        ,extrapolator=None)
    # execute
    res = bs.sample(driver,points)
    results_full[f'{bootstrap}'] =  res.raw_results
    results[f'points_{bootstrap}'] = res.points
    results[f'energies_{bootstrap}'] = res.energies

# define numpy solver
solver_numpy = NumPyMinimumEigensolver()
me_gsc_numpy = GroundStateEigensolver(ft, solver_numpy)
bs_classical = BOPESSampler(
               gss=me_gsc_numpy
               ,bootstrap=False
               ,num_bootstrap=None
               ,extrapolator=None)
# execute
res_np = bs_classical.sample(driver, points)
results_full['np'] =  res_np.raw_results
results['points_np'] = res_np.points
results['energies_np'] = res_np.energies


fig = plt.figure()
for value, bootstrap in conditions.items():
    plt.plot(results[f'points_{bootstrap}'], results[f'energies_{bootstrap}'], label = f'{bootstrap}')
plt.plot(results['points_np'], results['energies_np'], label = 'numpy')
plt.legend()
plt.title('Dissociation profile')
plt.xlabel('Interatomic distance')
plt.ylabel('Energy')

plt.show()