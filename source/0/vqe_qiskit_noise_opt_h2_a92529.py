# https://github.com/happybored/pennylane_vqe/blob/48fd6247693f783ccc0e7d033de469ab39a18f05/vqe/vqe_qiskit_noise_opt_H2.py
import numpy as np
import pylab
import copy
from qiskit import BasicAer,IBMQ
import qiskit
from qiskit.aqua import aqua_globals, QuantumInstance
from qiskit.aqua.algorithms import NumPyMinimumEigensolver
from myvqe import VQE
from qiskit.aqua.components.optimizers import SLSQP
from qiskit.algorithms.optimizers import SPSA
from qiskit.chemistry.components.initial_states import HartreeFock
from qiskit.chemistry.components.variational_forms import UCCSD,UVCC
from qiskit.chemistry.drivers import PySCFDriver
from qiskit.chemistry.core import Hamiltonian, QubitMappingType
import sys
from qiskit.ignis.mitigation import MeasurementFilter,CompleteMeasFitter
from qiskit import Aer
import os
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.test.mock import FakeToronto,FakeJakarta
from qiskit.circuit.library import EfficientSU2,TwoLocal
import math


vqe_energies = []
hf_energies = []
exact_energies = []
noisy_vqe_energies = []
noisy_mitigation_vqe_energies = []

compression_exact_energies = []
compression_noisy_vqe_energies = []
compression_noisy_mitigation_vqe_energies = []

original_length = []
compression_length =[]


dis = 1.3
molecule = 'Li .0 .0 -{0}; H .0 .0 {0}'.format(dis/2,dis/2)
# operator = Hamiltonian(qubit_mapping=QubitMappingType.PARITY, freeze_core=True,orbital_reduction=[-3,-2])

operator = Hamiltonian(qubit_mapping=QubitMappingType.PARITY,
                       two_qubit_reduction=True, freeze_core=True,
                       orbital_reduction=[-3,-2])
# def store_intermediate_result(eval_count, parameters, mean, std):
#     print(eval_count,mean)
#     exact_energies.append(mean)

# def store_intermediate_result_noise(eval_count, parameters, mean, std):
#     print(eval_count,mean)
#     noisy_vqe_energies.append(mean)

def get_fixing_parameters(weight,regu_val = math.pi *4 ):
    t1 = np.floor(weight/regu_val) 
    t2 = weight % (regu_val)
    t1[t2>regu_val/2] = t1[t2>regu_val/2] + 1
    t2[t2>regu_val/2] = regu_val/2 - t2[t2>regu_val/2]
    fixing_paras = t1*regu_val
    return fixing_paras


def get_fixing_abs(weight,fix_para):
    fixing_abs= np.abs(weight-fix_para)
    return fixing_abs

def compression(weight,percent=50):
    fixing_para = get_fixing_parameters(weight)
    weight_temp = get_fixing_abs(weight,fixing_para)
    percentile = np.percentile(weight_temp, percent)  # get a value for this percentitle
    under_threshold = weight_temp <= percentile
    weight[under_threshold] = fixing_para[under_threshold]
    return weight

def store_intermediate_result(vqe,eval_count, parameters, mean, std):
    # print('intermediate res:\n')
    print('='*50)
    print('step {}, original Energy:{}'.format(eval_count,mean))
    # compress
    cparameters =  copy.deepcopy(parameters)
    parameters1 = compression(cparameters)
    circuit1 = vqe.construct_circuit(parameters)[0]
    t1 =qiskit.transpile(circuit1,backend=device_backend)
    # print(t1)
    # print(len(t1))
    # print(parameters)
    original_length.append(len(t1))
    circuit2 = vqe.construct_circuit(parameters1)[0]
    t2 =qiskit.transpile(circuit2,backend=device_backend)
    # print('after compression:')
    # print(t2)
    # print(len(t2))
    # print(parameters1)
    compression_length.append(len(t2))
    means = vqe.manual_energy_evaluation(parameters1)
    print('step {}, after compression, Energy:{}'.format(eval_count,means))
    exact_energies.append(mean)
    compression_exact_energies.append(means)
    return parameters

def store_intermediate_result_noise(vqe,eval_count, parameters, mean, std):
    print('='*50)
    print('step {}, original Energy:{}'.format(eval_count,mean))
    # print(parameters)
    # compress
    cparameters =  copy.deepcopy(parameters)
    parameters2 = compression(cparameters)
    means = vqe.manual_energy_evaluation(parameters2)
    print('step {}, after compression, Energy:{}'.format(eval_count,means))
    noisy_vqe_energies.append(mean)
    compression_noisy_vqe_energies.append(means)
    return parameters

def store_intermediate_result_noise_mitigation(vqe,eval_count, parameters, mean, std):
    print('='*50)
    print('step {}, original Energy:{}'.format(eval_count,mean))
    cparameters = copy.deepcopy(parameters)
    parameters2 = compression(cparameters)
    means = vqe.manual_energy_evaluation(parameters2)
    print('step {}, after compression, Energy:{}'.format(eval_count,means))
    noisy_mitigation_vqe_energies.append(mean)
    compression_noisy_mitigation_vqe_energies.append(means)
    return parameters


backend = Aer.get_backend('aer_simulator')
counts1 = []
values1 = []
noise_model = None
device_backend = FakeJakarta()
seed =170


device = QasmSimulator.from_backend(device_backend)
coupling_map = device.configuration().coupling_map
noise_model = NoiseModel.from_backend(device)
basis_gates = noise_model.basis_gates

qi_state_vector = QuantumInstance(BasicAer.get_backend('statevector_simulator'))
qi_noise_mitigation = QuantumInstance(backend=backend, seed_simulator=seed, seed_transpiler=seed,
                     coupling_map=coupling_map, noise_model=noise_model,
                     measurement_error_mitigation_cls=CompleteMeasFitter,
                     cals_matrix_refresh_period=30)

qi_noise =QuantumInstance(backend, coupling_map=coupling_map, noise_model=noise_model)
print('noise_model:\n',noise_model)
print()



#setup

# molecule 2 molecule Hamiltonian
driver = PySCFDriver(molecule, basis='sto3g')
qmolecule = driver.run()

# Molecule Hamiltonian 2 Qubit Hamiltonian
# operator = Hamiltonian(qubit_mapping=QubitMappingType.PARITY)
qubit_op, aux_ops = operator.run(qmolecule)


#exact res HX = lambda*X
exact_result = NumPyMinimumEigensolver(qubit_op,aux_operators=aux_ops).run()
print('exact_result:\n',exact_result)
exact_result_mol = operator.process_algorithm_result(exact_result)
print('exact_result_mol:\n',exact_result_mol)


#VQE
# optimizer = SLSQP(maxiter=300,tol=1e-7)

optimizer = SPSA(maxiter=100)
# initial_state = HartreeFock(operator.molecule_info['num_orbitals'],
#                              operator.molecule_info['num_particles'],
#                              qubit_mapping=operator._qubit_mapping,
#                              two_qubit_reduction=operator._two_qubit_reduction)


# var_form = UCCSD(num_orbitals=operator.molecule_info['num_orbitals'],
#                  num_particles=operator.molecule_info['num_particles'],
#                  initial_state=initial_state,
#                  qubit_mapping=operator._qubit_mapping,
#                  two_qubit_reduction=operator._two_qubit_reduction)

# ——————————————————————————————————————————————————————————————————————

# var_form = EfficientSU2(qubit_op.num_qubits, entanglement="linear")
var_form = TwoLocal(qubit_op.num_qubits,['ry','rz'], 'cry', 'circular', reps=2, insert_barriers=True)

# print('initial_state:\n',initial_state)
print('var_form:\n',var_form)


algo = VQE(qubit_op, var_form, optimizer, aux_operators=aux_ops,callback=store_intermediate_result,max_evals_grouped=32)
vqe_result = algo.run(qi_state_vector)
print('vqe_result:\n',vqe_result)

vqe_result_molecule = operator.process_algorithm_result(vqe_result)
print('vqe_result_molecule:\n',vqe_result_molecule)



noisy_algo = VQE(qubit_op, var_form, optimizer, aux_operators=aux_ops, callback=store_intermediate_result_noise,max_evals_grouped=32)
noisy_vqe_result = noisy_algo.run(qi_noise)
print('noisy_vqe_result:\n',noisy_vqe_result)
noisy_vqe_result_molecule = operator.process_algorithm_result(noisy_vqe_result)
print('noisy_vqe_result_molecule:\n',noisy_vqe_result_molecule)



noisy_mitigation_algo = VQE(qubit_op, var_form, optimizer, aux_operators=aux_ops, callback=store_intermediate_result_noise_mitigation)
noisy_mitigation_vqe_result = noisy_mitigation_algo.run(qi_noise_mitigation)
print('noisy_mitigation_vqe_result:\n',noisy_mitigation_vqe_result)
noisy_mitigation_result_molecule = operator.process_algorithm_result(noisy_mitigation_vqe_result)
print('noisy_mitigation_result_molecule:\n',noisy_mitigation_result_molecule)

pylab.figure(1)

# pylab.plot(distances, hf_energies, label='Hartree-Fock')
# pylab.plot(distances, vqe_energies, 'o', label='vqe')
pylab.plot(range(0,len(exact_energies)), exact_energies,  label='Exact')
pylab.plot(range(0,len(noisy_vqe_energies)), noisy_vqe_energies, label='Noisy')
pylab.plot(range(0,len(noisy_mitigation_vqe_energies)), noisy_mitigation_vqe_energies, label='Mitigation')
pylab.plot(range(0,len(compression_exact_energies)), compression_exact_energies, label='Compression Exact')
pylab.plot(range(0,len(compression_noisy_vqe_energies)), compression_noisy_vqe_energies, label='Compression Noise')
pylab.plot(range(0,len(compression_noisy_mitigation_vqe_energies)), compression_noisy_mitigation_vqe_energies, label='Compression Mitigation')
print(np.real(exact_result['eigenvalue']))
constant = np.full(max(len(exact_energies),len(noisy_vqe_energies),len(noisy_mitigation_vqe_energies)), np.real(exact_result['eigenvalue']))
pylab.plot(range(0,len(constant)),constant , label='classic')
pylab.xlabel('Iter')
pylab.ylabel('Sum')
pylab.title('H-H Ground State Energy;distance ={}'.format(dis))
pylab.legend(loc='upper right')
pylab.savefig('energy.jpg')

# pylab.show()


pylab.figure(2)
print(original_length)
print(compression_length)

pylab.plot(range(0,len(original_length)), original_length,  label='original')
pylab.plot(range(0,len(compression_length)), compression_length,  label='compression')
pylab.title('H-H Ground State Energy;distance ={}'.format(dis))
pylab.legend(loc='upper right')
pylab.xlabel('Iter')
pylab.ylabel('length')
pylab.savefig('length.jpg')




