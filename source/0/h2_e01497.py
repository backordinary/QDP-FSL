# https://github.com/dfedorov1988/qiskit_dev_examples/blob/29844903d684d8ac0ae8ce8c0988616e36b0ec19/h2.py
import numpy as np
from qiskit.chemistry.drivers import PySCFDriver, UnitsType, Molecule
from qiskit.chemistry.transformations import FermionicTransformation, FermionicQubitMappingType
from qiskit.aqua.algorithms import NumPyMinimumEigensolver
from qiskit import Aer
from qiskit.aqua import QuantumInstance
from qiskit.chemistry.algorithms.ground_state_solvers.minimum_eigensolver_factories import VQEUCCSDFactory
from qiskit.chemistry.components.variational_forms import UCCSD
from qiskit.aqua.algorithms import VQE
from qiskit.chemistry.algorithms.ground_state_solvers import GroundStateEigensolver
from qiskit.aqua.operators import I, X, Z
from qiskit.chemistry.circuit.library.initial_states import HartreeFock

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

def compute_gradient(distance):


    def construct_qubit_ham(distance):


        molecule = Molecule(geometry=[['H', [0., 0., 0.]],
                                    ['H', [0., 0., distance]]],
                                    charge=0, multiplicity=1)
        driver = PySCFDriver(molecule = molecule, unit=UnitsType.ANGSTROM, basis='sto3g')
        transformation = FermionicTransformation(qubit_mapping=FermionicQubitMappingType.PARITY, two_qubit_reduction=False)
        qubit_ham = transformation.transform(driver)
        e_nr = transformation._nuclear_repulsion_energy
        return driver, transformation, qubit_ham, e_nr


    # distance = 0.735
    dx = 1e-10
    driver, transformation, qubit_ham, e_nr = construct_qubit_ham(distance)
    driver_mdx, transformation_mdx, qubit_ham_mdx, e_nr_mdx = construct_qubit_ham(distance-dx)
    driver_pdx, transformation_pdx, qubit_ham_pdx, e_nr_pdx = construct_qubit_ham(distance+dx)
    # print("Rep energies", e_nr_mdx, e_nr_pdx)

    num_orbitals = transformation.molecule_info['num_orbitals']
    num_particles = transformation.molecule_info['num_particles']
    qubit_mapping = transformation.qubit_mapping
    two_qubit_reduction = transformation.molecule_info['two_qubit_reduction']
    z2_symmetries = transformation.molecule_info['z2_symmetries']

    numpy_solver = NumPyMinimumEigensolver()
    calc = GroundStateEigensolver(transformation, numpy_solver)
    res_mdx = calc.solve(driver_mdx)
    res_pdx = calc.solve(driver_pdx)
    grad = (res_pdx['computed_energies'] + e_nr_pdx - res_mdx['computed_energies'] - e_nr_mdx) / 2 / dx
    print("grad", grad) 


    # initial_state = HartreeFock(num_orbitals, num_particles, qubit_mapping,
    #                                     two_qubit_reduction, z2_symmetries.sq_list)

    # var_form = UCCSD(num_orbitals=num_orbitals,
    #                 num_particles=num_particles,
    #                 initial_state=initial_state,
    #                 qubit_mapping=qubit_mapping,
    #                 two_qubit_reduction=two_qubit_reduction,
    #                 z2_symmetries=z2_symmetries)

    # vqe = VQE(var_form=var_form, quantum_instance=QuantumInstance(Aer.get_backend('statevector_simulator')))
    # aux_operators=[qubit_ham_mdx[0], qubit_ham_pdx[0]]
    # vqe._aux_operators = aux_operators
    # vqe._operator = qubit_ham[0]
    # res = vqe._run()
    # grad = (vqe._ret['aux_ops'][1]+e_nr_pdx - vqe._ret['aux_ops'][0]-e_nr_mdx)/dx/2
    # print('gradient', grad)

    # calc = GroundStateEigensolver(transformation, vqe)
    # res = calc.solve(driver)
    # print(res)

 
for n in range(100):
    distance = 0.73486 + 0.000001*n
    print("r=", distance)
    compute_gradient(distance) 