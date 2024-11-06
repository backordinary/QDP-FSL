# https://github.com/MartenSkogh/QiskitVQEWrapper/blob/5f1e3e84769e4978e0b50b30e3e48873d7dd4f41/vqe_wrapper/VQE_wrapper.py
import sys
import numpy as np
import scipy as sp
import re
from copy import deepcopy
from pprint import pprint
from timeit import default_timer as timer
from enum import Enum

from qiskit import Aer
from qiskit.aqua import QuantumInstance
from qiskit.aqua.operators import Z2Symmetries
from qiskit.aqua.algorithms.minimum_eigen_solvers import VQE
from qiskit.aqua.algorithms import ExactEigensolver
from qiskit.aqua.components.optimizers import SLSQP, L_BFGS_B, COBYLA, SPSA
from qiskit.chemistry.core import Hamiltonian, TransformationType, QubitMappingType
from qiskit.chemistry.drivers import PySCFDriver, GaussianDriver, UnitsType, HFMethodType
from qiskit.chemistry.components.variational_forms import UCCSD 
from qiskit.chemistry.components.initial_states import HartreeFock

class DriverType(Enum):
    """ DriverType Enum """
    PYSCF = 'PySCF'
    GAUSSIAN = 'Gaussian'

class VQEWrapper():
    
    def __init__(self):

        ### MOLECULE ###
        # These things need to be set before running
        self.molecule_string = None
        # You can make a pretty educated guess for these two
        self.spin = None
        self.charge = None

        self.qmolecule = None

        ### CHEMISTRY DRIVER ###
        #Basis has to be in a format accepted by Gaussian (sto-3g, 6-31g)
        self.basis = 'sto-3g'
        self.chem_driver = DriverType.GAUSSIAN
        self.hf_method = HFMethodType.UHF
        self.length_unit = UnitsType.ANGSTROM
        self.gaussian_checkfile = ''
        
        self.driver = None
        self.core = None

        ### HAMILTONIAN ###
        self.transformation = TransformationType.FULL
        self.qubit_mapping = QubitMappingType.JORDAN_WIGNER
        self.two_qubit_reduction = False
        self.freeze_core = True
        self.orbital_reduction = []

        self.qubit_op = None
        self.aux_ops = None
        self.initial_point = None

        self.optimizer = SLSQP(maxiter=5000)

        self.ansatz = 'UCCSD'
        self.excitation_type = 'sd'
        self.num_time_slices = 1
        self.shallow_circuit_concat = False
        
        self.vqe_algo = None

        self.var_form = None
        self.vqe_callback = None
        self.vqe_time = None


        ### BACKEND CONFIG ###
        #Choose the backend (use Aer instead of BasicAer) 
        self.simulator = 'statevector_simulator'
        self.shots = 1024
        self.seed_simulator = None
        self.seed_transpiler = None
        self.noise_model = None
        self.measurement_error_mitigation_cls = None
        self.backend_options = {}


    def opt_str(self):
        match = re.search(r'optimizers.[A-z]+.(.+) object', str(self.optimizer))
        opt_str = match.group(1)
        return opt_str

    def gaussian_config(self):
        #Format properties to a string fitting the gaussian input format
        if self.gaussian_checkfile != '':
            chk = f'%Chk={self.gaussian_checkfile}\n'
        else:
            chk = ''
        gaussian_config = chk + f'# {self.hf_method.value}/{self.basis} scf(conventional)\n\nMolecule\n\n{self.charge} {self.spin+1}\n'
        gaussian_config = gaussian_config + self.molecule_string.replace('; ','\n') + '\n\n'
        return gaussian_config

    def initiate(self):

        self.init_backend()
        self.init_driver()
        self.init_core()
        self.init_ops()
        self.init_init_state()
        self.init_var_form()
        self.init_vqe()

    def init_driver(self):

        if self.chem_driver.value == 'PySCF':
            if self.hf_method == HFMethodType.RHF and self.spin % 2 == 0:
                print(f'WARNING: Restricted Hartree-Fock (RHF) cannot handle unpaired electrons!')
                print(f'Switching to Unrestricted Hartree-Fock!')
                self.chem_driver = HFMethodType.UHF

            self.driver = PySCFDriver(atom=self.molecule_string, 
                                      unit=self.length_unit, 
                                      charge=self.charge,
                                      spin=self.spin,
                                      hf_method=self.hf_method,
                                      basis=self.basis)

        elif self.chem_driver.value == 'Gaussian':
            self.driver = GaussianDriver(config=self.gaussian_config())

        self.qmolecule = self.driver.run()
        

    def init_backend(self):
        self.backend = Aer.get_backend(self.simulator) 
        self.quantum_instance = QuantumInstance(backend=self.backend,
                                                shots=self.shots,
                                                seed_simulator = self.seed_simulator,
                                                seed_transpiler = self.seed_transpiler,
                                                noise_model = self.noise_model,
                                                measurement_error_mitigation_cls = self.measurement_error_mitigation_cls,
                                                backend_options = self.backend_options)

    def init_core(self):
        self.core = Hamiltonian(transformation=self.transformation, 
                                qubit_mapping=self.qubit_mapping, 
                                two_qubit_reduction=self.two_qubit_reduction, 
                                freeze_core=self.freeze_core, 
                                orbital_reduction=self.orbital_reduction)

    def init_ops(self):
        self.qubit_op, self.aux_ops = self.core.run(self.qmolecule)


    #Initial state
    def init_init_state(self):
        self.init_state = HartreeFock(num_orbitals=self.core._molecule_info['num_orbitals'], 
                                      qubit_mapping=self.core._qubit_mapping,
                                      two_qubit_reduction=self.core._two_qubit_reduction, 
                                      num_particles=self.core._molecule_info['num_particles'])


    #Set up VQE
    def init_vqe(self):
        self.vqe_algo = VQE(self.qubit_op, 
                            self.var_form, 
                            self.optimizer, 
                            initial_point=self.initial_point, 
                            callback=self.vqe_callback)


    def init_var_form(self):
        if self.ansatz.upper() == 'UCCSD':
            # UCCSD Ansatz
            self.var_form = UCCSD(num_orbitals=self.core._molecule_info['num_orbitals'], 
                                  num_particles=self.core._molecule_info['num_particles'], 
                                  initial_state=self.init_state, 
                                  qubit_mapping=self.core._qubit_mapping, 
                                  two_qubit_reduction=self.core._two_qubit_reduction, 
                                  num_time_slices=self.num_time_slices, 
                                  excitation_type=self.excitation_type,
                                  shallow_circuit_concat=self.shallow_circuit_concat)
        else:
            if self.var_form is None:
                raise ValueError('No variational form specified!')
                
    def print_config(self):
        print(f'\n\n=== MOLECULAR INFORMATION ===')
        print(f'*  Molecule string: {self.molecule_string}')
        print(f'*  Charge: {self.charge}')
        print(f'*  Spin (2S): {self.spin}')
        print(f'*  Basis set: {self.basis}')
        print(f'*  Num orbitals: {self.qmolecule.num_orbitals}')
        print(f'*  Lenght Unit: {self.length_unit}')
        print(f'*  HF method: {self.hf_method}')
        
        print(f'\n\n=== HAMILTONIAN INFORMATION ===')
        print(f'*  Transformation type: {self.transformation}')
        print(f'*  Qubit mapping: {self.qubit_mapping}')
        print(f'*  Two qubit reduction: {self.two_qubit_reduction}')
        print(f'*  Freeze core: {self.freeze_core}')
        print(f'*  Orbital reduction: {self.orbital_reduction}')

        print(f'\n\n=== CHEMISTRY DRIVER INFORMATION ===')
        print(f'*  Not yet implemented!')

        print(f'\n\n=== BACKEND INFORMATION ===')
        print(f'*  Not yet implemented!')

    def run_vqe(self):
        #Run the algorithm
        vqe_start = timer()
        self.vqe_result = self.vqe_algo.run(self.quantum_instance)
        self.vqe_time = timer() - vqe_start

        #Get the results
        result = self.core.process_algorithm_result(self.vqe_result) 

        return result
