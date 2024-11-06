# https://github.com/HappyBravo/QIQT_22/blob/0f52a62fdb373576633e1b3c9389313dd18ba094/Tutorial5/h2.py
##############################################
######## THIS DOES NOT WORK ON WINDOWS #######
### BECAUSE OF pyscf - Chemistry Framework ###
##############################################


from qiskit_nature.drivers.second_quantization import PySCFDriver
from qiskit_nature.problems.second_quantization.electronic import ElectronicStructureProblem

from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper
from qiskit_nature.circuit.library import HartreeFock, UCC
from qiskit.algorithms.optimizers import COBYLA, CG, SLSQP, L_BFGS_B
from qiskit.algorithms import VQE
from qiskit_nature.algorithms.ground_state_solvers.minimum_eigensolver_factories import NumPyMinimumEigensolverFactory
from qiskit import Aer
import numpy as np 
import matplotlib.pyplot as plt
from qiskit_nature.algorithms.ground_state_solvers import GroundStateEigensolver
from qiskit.circuit import Parameter, QuantumCircuit, QuantumRegister

# import logging
# logging.basicConfig(filename="H2_1.0.log", level=logging.INFO)
# logger=logging.getLogger()
# logger.setLevel(logging.DEBUG)

def store_intermediate_result(eval_count, parameters, mean, std):
    counts.append(eval_count)
    values.append(mean)

h = np.linspace(1, 2, 20)
FCI_energy = []
HF_energy = []
UCCSD_energy = []
for i in h:
    coord = f'Li 0.0 0.0 0.0; H 0.0 0.0 {i}' #	f'H 0.0 0.0 0.0; H 0.0 0.0 {i}'
    print(f"Finding for {coord}")
    driver = PySCFDriver(atom=coord, charge=0, spin=0, basis='sto3g')
    es_problem = ElectronicStructureProblem(driver)

    # obtaining qubit Hamiltonian
    mapper = JordanWignerMapper()
    converter = QubitConverter(mapper=mapper, two_qubit_reduction=False)
    second_q_op = es_problem.second_q_ops()
    # print(second_q_op['ElectronicEnergy'])
    qubit_op = converter.convert(second_q_op['ElectronicEnergy'])
    # print(qubit_op)

    es_particle_number = es_problem.grouped_property_transformed.get_property('ParticleNumber')
    num_particles = (es_particle_number.num_alpha, es_particle_number.num_beta)
    num_spin_orbitals = es_particle_number.num_spin_orbitals
    es_energy = es_problem.grouped_property_transformed.get_property('ElectronicEnergy')
    #print(es_energy.electronic_energy)
    nuclear_repulsion_energy = es_energy.nuclear_repulsion_energy
    shift = nuclear_repulsion_energy
    # print('Number Of Particles: ',num_particles)
    # print('Number of Spin Orbitals: ', num_spin_orbitals)
    # print('Nuclear Repulsion Energy: ', nuclear_repulsion_energy)


    # initialization of state
    init_state = HartreeFock(num_spin_orbitals, num_particles, converter)

    #optimizer = L_BFGS_B(maxfun=200000,maxiter=10000)
    backend = Aer.get_backend('statevector_simulator')

    #Create dummy parametrized circuit for HF calculation
    theta = Parameter('a')
    n = qubit_op.num_qubits
    qc = QuantumCircuit(qubit_op.num_qubits)
    qc.rz(theta*0,0)
    ansatz = qc
    ansatz.compose(init_state, front=True, inplace=True)

    #Pass it through VQE
    algorithm = VQE(ansatz,quantum_instance=backend)
    result = algorithm.compute_minimum_eigenvalue(qubit_op).eigenvalue
    # print('HF Energy is',np.real(result)+shift)
    HF_energy.append(np.real(result)+shift)

    # Variational ansatz formation
    var_form = UCC(num_particles=num_particles,num_spin_orbitals=num_spin_orbitals, excitations='sd', initial_state=init_state, qubit_converter=converter)
    excitation_list = var_form._get_excitation_list()
    # print('no of parameters',var_form.num_parameters)
    # print('Excitation list is',excitation_list)
    circuit = var_form.decompose() #.decompose().decompose()
    # print(circuit)


    backend = Aer.get_backend('statevector_simulator')

    optimizer = COBYLA(maxiter=10000)

    counts = list()
    values = list()
    
    #VQE optimization
    vqe1 = VQE(var_form, optimizer=optimizer, callback = store_intermediate_result, quantum_instance=backend)
    vqe_result = vqe1.compute_minimum_eigenvalue(qubit_op)
    E1 = np.real(vqe_result.eigenvalue)+shift
    # print('VQE Optimized UCCSD Energy is',E1)
    UCCSD_energy.append(E1)

    # NumPyMinimumEigensolver (FCI Energy)
    solver = NumPyMinimumEigensolverFactory()
    calc = GroundStateEigensolver(converter, solver)
    numpy_result = calc.solve(es_problem)
    exact_energy = np.real(numpy_result.eigenenergies[0]) + shift
    
    FCI_energy.append(exact_energy)

# print(zip(h, energy))
plt.plot(h, HF_energy, label = "HF")
plt.plot(h, UCCSD_energy, label = "UCCSD")
plt.plot(h, FCI_energy, label = "FCI")
plt.grid()
plt.legend()

plt.show()
