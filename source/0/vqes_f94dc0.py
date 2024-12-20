# https://github.com/rodolfocarobene/SPVQE/blob/550ab1282cbf1e9bc5665acaca8b2ecd34868b74/spvqe/vqes.py
import logging
from datetime import datetime

import math
import numpy as np

from qiskit import IBMQ
from qiskit import Aer
from qiskit import QuantumCircuit

from qiskit.algorithms.optimizers import L_BFGS_B, COBYLA
from qiskit.algorithms import VQE
from qiskit.circuit import QuantumRegister, Parameter
from qiskit.circuit.library import TwoLocal, EfficientSU2
from qiskit.utils import QuantumInstance
from qiskit.opflow.primitive_ops import PauliOp
from qiskit.quantum_info import Pauli
from qiskit.providers.ibmq import least_busy

from qiskit_nature.drivers import UnitsType, Molecule
from qiskit_nature.drivers.second_quantization import PySCFDriver
from qiskit_nature.drivers.second_quantization.electronic_structure_driver import MethodType
from qiskit_nature.problems.second_quantization.electronic import ElectronicStructureProblem
from qiskit_nature.results import ElectronicStructureResult
from qiskit_nature.mappers.second_quantization import ParityMapper, BravyiKitaevMapper
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.transformers.second_quantization.electronic import FreezeCoreTransformer, ActiveSpaceTransformer
from qiskit_nature.circuit.library import HartreeFock, UCCSD
from qiskit_nature.algorithms import VQEUCCFactory, GroundStateEigensolver
from qiskit_nature.results import EigenstateResult
from qiskit_nature.runtime import VQEProgram

def order_of_magnitude(number):
    if number == 0:
        return -100
    return math.floor(math.log(number, 10))

def get_date_time_string():
    now = datetime.now()
    return now.strftime("%d_%m_%H_%M")

myLogger = logging.getLogger('myLogger')
myLogger.setLevel(logging.DEBUG)
ch = logging.FileHandler('./logs/' + get_date_time_string() + '.log')
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
myLogger.addHandler(ch)

def add_single_so4_gate(circuit,
                     qubit1,
                     qubit2,
                     params,
                     par0):
    myLogger.info('Inizio di add_single_so4_gate')

    circuit.s(qubit1)
    circuit.s(qubit2)
    circuit.h(qubit2)
    circuit.cx(qubit2, qubit1)
    circuit.u(params[par0], params[par0+1], params[par0+2], qubit1)
    par0 += 3
    circuit.u(params[par0], params[par0+1], params[par0+2], qubit2)
    par0 += 3
    circuit.cx(qubit2, qubit1)
    circuit.h(qubit2)
    circuit.sdg(qubit1)
    circuit.sdg(qubit2)

    myLogger.info('Fine di add_single_so4_gate')

def construct_so4_ansatz(numqubits, init=None):
    myLogger.info('Inizio di construct_so4_ansatz')

    num_parameters = 6 * (numqubits - 1)
    so4_parameters = []
    for i in range(num_parameters):
        name = "par" + str(i)
        new = Parameter(name)
        so4_parameters.append(new)
    quantum_reg = QuantumRegister(numqubits, name='q')
    if init is not None:
        circ = init
    else:
        circ = QuantumCircuit(quantum_reg)
    i = 0
    j = 0
    while i + 1 < numqubits:
        add_single_so4_gate(circ, i, i+1, so4_parameters, 6*j)
        j = j + 1
        i = i + 2
    i = 1
    while i + 1 < numqubits:
        add_single_so4_gate(circ, i, i+1, so4_parameters, 6*j)
        j = j +1
        i = i + 2

    myLogger.info('Fine di construct_so4_ansatz')

    return circ

def create_lagrange_operator_ps(hamiltonian,
                             auxiliary,
                             multiplier,
                             operator,
                             value):
    myLogger.info('Inizio di create_lagrange_operator_ps')

    if operator == "number":
        idx = 0
    elif operator == "spin-squared":
        idx = 1
    elif operator == "spin-z":
        idx = 2

    list_x_zeros = np.zeros(hamiltonian.num_qubits)
    list_z_zeros = np.zeros(hamiltonian.num_qubits)

    equality = auxiliary[idx].add(PauliOp(Pauli((list_z_zeros,
                                                 list_x_zeros)),
                                          -value))

    penalty_squared = (equality ** 2).mul(multiplier)

    lagrangian = hamiltonian.add(penalty_squared)

    myLogger.info('Fine di create_lagrange_operator_ps')

    return lagrangian

def create_lagrange_operator_aug(hamiltonian,
                             auxiliary,
                             multiplier_simple,
                             multiplier_square,
                             operator,
                             value):
    myLogger.info('Inizio di create_lagrange_operator_aug')

    if operator == "number":
        idx = 0
    elif operator == "spin-squared":
        idx = 1
    elif operator == "spin-z":
        idx = 2

    list_x_zeros = np.zeros(hamiltonian.num_qubits)
    list_z_zeros = np.zeros(hamiltonian.num_qubits)

    equality = auxiliary[idx].add(PauliOp(Pauli((list_z_zeros,
                                                 list_x_zeros)),
                                          -value))

    penalty_squared = (equality ** 2).mul(multiplier_square)
    penalty_simple = equality.mul(-multiplier_simple)

    lagrangian = hamiltonian.add(penalty_squared).add(penalty_simple)

    myLogger.info('Fine di create_lagrange_operator_aug')

    return lagrangian

def get_transformers_from_mol_type(mol_type):
    transf_list = []
    if mol_type == 'LiH':
        #transf_list.append(FreezeCoreTransformer(True)) #, [2, 3, 4]))
        transf_list.append(ActiveSpaceTransformer(num_electrons=2,
                                                  num_molecular_orbitals=3))
    if mol_type == 'H2O':
        transf_list.append(ActiveSpaceTransformer(num_electrons=4,
                                                  num_molecular_orbitals=3))

    if mol_type == 'C2H4':
        transf_list.append(ActiveSpaceTransformer(num_electrons=2,
                                                  num_molecular_orbitals=2))
    if mol_type == 'N2':
        transf_list.append(ActiveSpaceTransformer(num_electrons=6,
                                                  num_molecular_orbitals=3))
    if mol_type == 'Li3+':
        transf_list.append(ActiveSpaceTransformer(num_electrons=2,
                                                  num_molecular_orbitals=3))
    if mol_type == 'Na-':
        transf_list.append(FreezeCoreTransformer(True))
    return transf_list

def from_geometry_to_atoms(geometry):
    tot_atoms = []
    atoms_geom = geometry.split(';')
    for single_atom_geom in atoms_geom:
        atom = single_atom_geom.split()[0]
        tot_atoms.append(atom)
    return tot_atoms

def get_num_particles(mol_type,
                      particle_number):

    alpha, beta = particle_number.num_alpha, particle_number.num_beta
    num_spin_orbitals = particle_number.num_spin_orbitals

    if mol_type == 'LiH':
        a_b_spinorbs = 1, 1, 6 #10#4
    if mol_type == 'Li3+':
        a_b_spinorbs = 1, 1, 6 #10#4
    elif mol_type == 'H2O':
        a_b_spinorbs = 2, 2, 6
    elif mol_type == 'C2H4':
        a_b_spinorbs = 1, 1, 4
    elif mol_type == 'N2':
        a_b_spinorbs = 3, 3, 6
    else:
        a_b_spinorbs = alpha, beta, num_spin_orbitals

    return a_b_spinorbs

def prepare_base_vqe(options):
    myLogger.info('Inizio di prepare_base_vqe')

    spin = options['molecule']['spin']
    charge = options['molecule']['charge']
    basis = options['molecule']['basis']
    geometry = options['molecule']['geometry']

    var_form_type = options['var_form_type']
    quantum_instance = options['quantum_instance']
    optimizer = options['optimizer']
    converter = options['converter']
    init_point = options['init_point']

    driver = PySCFDriver(atom=geometry,
                         unit=UnitsType.ANGSTROM,
                         basis=basis,
                         spin=spin,
                         charge=charge,
                         method=MethodType.RHF)

    transformers = get_transformers_from_mol_type(options['molecule']['molecule'])
    problem = ElectronicStructureProblem(driver, transformers)
    main_op = problem.second_q_ops()[0]

    driver_result = driver.run()
    particle_number = driver_result.get_property('ParticleNumber')

    alpha, beta, num_spin_orbitals = get_num_particles(options['molecule']['molecule'],
                                                       particle_number)


    num_particles = (alpha, beta)

    myLogger.info("alpha %d", alpha)
    myLogger.info("beta %d", beta)
    myLogger.info("spin-orb %d", num_spin_orbitals)

    qubit_op = converter.convert(main_op, num_particles=num_particles)

    init_state = HartreeFock(num_spin_orbitals,
                             num_particles,
                             converter)

    num_qubits = qubit_op.num_qubits

    myLogger.info("num qubit qubitop : %d", qubit_op.num_qubits)
    if init_state.num_qubits != num_qubits:
        myLogger.info("num qubit initsta: %d", init_state.num_qubits)
        init_state = None

    vqe_solver = create_vqe_from_ansatz_type(var_form_type,
                                         num_qubits,
                                         init_state,
                                         quantum_instance,
                                         optimizer,
                                         converter,
                                         num_particles,
                                         num_spin_orbitals,
                                         init_point)

    myLogger.info('Fine di prepare_base_vqe')

    return converter, vqe_solver, problem, qubit_op

def store_intermediate_result(count, par, energy, std):
    global PARAMETERS
    PARAMETERS.append(par)
    log_string = str(count) + ' ' + str(energy) + ' ' + str(std)
    myLogger.info(log_string)

def from_energy_pars_to_log_msg(pars, energy):
    message = ''
    for par in pars:
        message += str(par).strip()
        message += ','
    message = message[:-1]
    message += ' ; '
    message += str(energy)
    message += '\n'
    return message

def get_ansatz(var_form_type, num_qubits, init_state=None):
    if var_form_type == 'TwoLocal':
        ansatz = TwoLocal(num_qubits=num_qubits,
                          rotation_blocks='ry',
                          entanglement_blocks='cx',
                          initial_state=init_state,
                          entanglement='linear')

    elif var_form_type == 'EfficientSU(2)':
        ansatz = EfficientSU2(num_qubits=num_qubits,
                              entanglement='linear',
                              reps=1,
                              skip_final_rotation_layer=True,
                              initial_state=init_state)
    else:
        print('Ansatz non ancora implementato in get_ansatz()')

    return ansatz


def create_vqe_from_ansatz_type(var_form_type,
                            num_qubits,
                            init_state,
                            quantum_instance,
                            optimizer,
                            converter,
                            num_particles,
                            num_spin_orbitals,
                            initial_point):
    myLogger.info('Inizio create_vqe_from_ansatz_type')

    if var_form_type in ('TwoLocal', 'EfficientSU(2)'):
        ansatz = get_ansatz(var_form_type, num_qubits, init_state)
        if None in initial_point:
            initial_point = np.random.rand(ansatz.num_parameters)
        vqe_solver = VQE(ansatz=ansatz,
                         optimizer=optimizer,
                         initial_point=initial_point,
                         callback=store_intermediate_result,
                         quantum_instance=quantum_instance)

    elif var_form_type == 'UCCSD':

        ansatz = UCCSD(qubit_converter=converter,
                       initial_state=init_state,
                       num_particles=num_particles,
                       num_spin_orbitals=num_spin_orbitals)

        if None in initial_point:
            initial_point = np.random.rand(8) #MUST BE SET HERE

        vqe_solver = VQE(quantum_instance=quantum_instance,
                         ansatz=ansatz._build(),
                         optimizer=optimizer,
                         callback=store_intermediate_result,
                         initial_point=initial_point)

    elif var_form_type == 'SO(4)':
        ansatz = construct_so4_ansatz(num_qubits,
                                    init=init_state)
        if None in initial_point:
            initial_point = np.random.rand(6*(num_qubits-1))
        vqe_solver = VQE(ansatz=ansatz,
                         optimizer=optimizer,
                         initial_point=initial_point,
                         callback=store_intermediate_result,
                         quantum_instance=quantum_instance)

    else:
        raise Exception("VAR_FORM_TYPE NOT EXISTS")

    myLogger.info('Fine  create_vqe_from_ansatz_type')

    return vqe_solver

def solve_hamiltonian_vqe(options):
    myLogger.info('Inizio solve_hamiltonian_vqe')
    myLogger.info('OPTIONS')
    myLogger.info(options)

    converter, vqe_solver, problem, qubit_op = prepare_base_vqe(options)

    calc = GroundStateEigensolver(converter, vqe_solver)
    result = calc.solve(problem)

    myLogger.info('Fine solve_hamiltonian_vqe')
    myLogger.info(PARAMETERS[len(PARAMETERS)-1])
    myLogger.info('RESULT')
    myLogger.info(result)

    return result

def get_runtime_vqe_program(options, num_qubits):
    #IBMQ.load_account()
    provider = IBMQ.get_provider(hub='ibm-q-research-2', group='uni-milano-bicoc-1', project='main')

    backend_list = []

    backend_list.append(provider.get_backend('ibm_perth'))
    backend_list.append(provider.get_backend('ibm_lagos'))
    backend_list.append(provider.get_backend('ibmq_casablanca'))
    backend_list.append(provider.get_backend('ibmq_bogota'))
    backend_list.append(provider.get_backend('ibmq_manila'))

    quantum_instance = least_busy(backend_list)

    print('Run su backend: ', quantum_instance.backend_name)

    ansatz = get_ansatz(options['var_form_type'], num_qubits)

    init_point = options['init_point']
    if None in init_point:
        init_point = np.random.rand(ansatz.num_parameters)

    global LAST_OPTIMIZER_OPT
    if LAST_OPTIMIZER_OPT is not None:
        options['optimizer'].set_options(LAST_OPTIMIZER_OPT)

    vqe_program = VQEProgram(
        ansatz=ansatz,
        optimizer=options['optimizer'],
        initial_point=init_point,
        provider=provider,
        backend=quantum_instance,
        shots=options['shots'],
        callback=store_intermediate_result,
        store_intermediate=True
    )

    return vqe_program

def solve_lagrangian_vqe(options):
    myLogger.info('Inizio solve_lagrangian_vqe')
    myLogger.info('OPTIONS')
    myLogger.info(options)

    converter, vqe_solver, problem, qubit_op = prepare_base_vqe(options)

    if options['hardware'] is True:
        vqe_solver = get_runtime_vqe_program(options, qubit_op.num_qubits)

    aux_ops_not_converted = problem.second_q_ops()[1:4]
    aux_ops = convert_list_op_ferm_to_qubit(aux_ops_not_converted,
                                         converter,
                                         problem.num_particles)

    lagrange_op = qubit_op
    for operatore in options['lagrange']['operators']:
        operator = operatore[0]
        value = operatore[1]
        multiplier = operatore[2]
        lagrange_op = create_lagrange_operator_ps(lagrange_op,
                                               aux_ops,
                                               multiplier=multiplier,
                                               operator=operator,
                                               value=value)

    old_result = vqe_solver.compute_minimum_eigenvalue(operator=lagrange_op,
                                                       aux_operators=aux_ops)

    global PARAMETERS
    if options['hardware'] is True:
        for elem in old_result.optimizer_history['params']:
            PARAMETERS.append(list(elem))

    myLogger.info('OLDRESULT:')
    myLogger.info(old_result)
    new_result = problem.interpret(old_result)

    myLogger.info('Fine solve_lagrangian_vqe')
    myLogger.info('RESULT')
    myLogger.info(new_result)

    if options['hardware'] is True:
        LAST_OPTIMIZER_OPT = options['optimizer'].setting

    myLogger.info('OPTIMIZER SETTINGS:')
    myLogger.info(options['optimizer'].setting)

    return new_result

def solve_aug_lagrangian_vqe(options, lamb):
    myLogger.info('Inizio solve_aug_lagrangian_vqe')
    myLogger.info('OPTIONS')
    myLogger.info(options)

    converter, vqe_solver, problem, qubit_op = prepare_base_vqe(options)

    if options['var_form_type'] == 'UCCSD':
        vqe_solver = vqe_solver.get_solver(problem, converter)

    aux_ops_not_converted = problem.second_q_ops()[1:4]
    aux_ops = convert_list_op_ferm_to_qubit(aux_ops_not_converted,
                                         converter,
                                         problem.num_particles)

    lagrange_op = qubit_op
    for operatore in options['lagrange']['operators']:
        operator = operatore[0]
        value = operatore[1]
        multiplier = operatore[2]
        lagrange_op = create_lagrange_operator_aug(lagrange_op,
                                                aux_ops,
                                                multiplier_square=multiplier,
                                                multiplier_simple=lamb,
                                                operator=operator,
                                                value=value)

    old_result = vqe_solver.compute_minimum_eigenvalue(operator=lagrange_op,
                                                      aux_operators=aux_ops)

    myLogger.info('OLDRESULT:')
    myLogger.info(old_result)
    new_result = problem.interpret(old_result)

    myLogger.info('Fine solve_aug_lagrangian_vqe')
    myLogger.info('RESULT')
    myLogger.info(new_result)

    return new_result

def convert_list_op_ferm_to_qubit(old_aux_ops, converter, num_particles):
    myLogger.info('Inizio convert_list_op_ferm_to_qubit')

    new_aux_ops = []
    for old_aux_op in old_aux_ops:
        op_new = converter.convert(old_aux_op, num_particles)
        new_aux_ops.append(op_new)

    myLogger.info('Fine convert_list_op_ferm_to_qubit')

    return new_aux_ops

def find_best_result(partial_results):
    penal_min = 100
    optimal_par = []

    tmp_result = ElectronicStructureResult()
    tmp_result.nuclear_repulsion_energy = 50
    tmp_result.computed_energies = np.array([0])
    tmp_result.extracted_transformer_energies = {'dummy': 0}

    for result, penalty, par in partial_results:
        myLogger.info('currE: %s', str(tmp_result.total_energies[0]))
        myLogger.info('GUARDO: %s', str(penalty))
        myLogger.info('CONFRONTO: %s', str(penal_min))

        if abs(order_of_magnitude(penalty) - order_of_magnitude(penal_min)) == 0:
            if tmp_result.total_energies[0] > result.total_energies[0]:
                tmp_result = result
                penal_min = penalty
                optimal_par = par
        elif penalty < penal_min:
            tmp_result = result
            penal_min = penalty
            optimal_par = par

        myLogger.info('newE: %s', str(tmp_result.total_energies[0]))

    return tmp_result, optimal_par

def calc_penalty(lag_op_list, result, threshold, tmp_mult):
    penalty = 0
    accectable_result = True

    for operatore in lag_op_list:
        if operatore[0] == 'number':
            penalty += tmp_mult*((result.num_particles[0] - operatore[1])**2)
            myLogger.info('penalty at number: %s', str(penalty))
            if abs(result.num_particles[0] - operatore[1]) > threshold:
                accectable_result = False
        if operatore[0] == 'spin-squared':
            penalty += tmp_mult*((result.total_angular_momentum[0] - operatore[1])**2)
            myLogger.info('penalty at spin2: %s', str(penalty))
            if abs(result.total_angular_momentum[0] - operatore[1]) > threshold:
                accectable_result = False
        if operatore[0] == 'spin-z':
            penalty += tmp_mult*((result.magnetization[0] - operatore[1])**2)
            myLogger.info('penalty at spinz: %s', str(penalty))
            if abs(result.magnetization[0] - operatore[1]) > threshold:
                    accectable_result = False

    return penalty, accectable_result

def solve_lag_series_vqe(options):
    iter_max = options['series']['itermax']
    step = options['series']['step']

    if 'init_point' not in options:
        par = np.random.rand(get_num_par(options['var_form_type'],
                                         options['molecule']['molecule']))
    else:
        par = options['init_point']

    mult = 0.01
    threshold = 0.6

    global PARAMETERS
    PARAMETERS = [par]

    partial_results = []

    for i in range(iter_max):
        tmp_mult = mult + step * i

        lag_op_list = []

        for single_op in options['lagrange']['operators']:
            operatore = (single_op[0],
                         single_op[1],
                         float(tmp_mult))
            lag_op_list.append(operatore)

        options['lagrange']['operators'] = lag_op_list

        options['init_point'] = par
        result = solve_lagrangian_vqe(options)

        if options['hardware'] is not True:
            par = PARAMETERS[len(PARAMETERS) - 1]

        penalty, accectable_result = calc_penalty(lag_op_list,
                                                  result,
                                                  threshold,
                                                  tmp_mult)

        log_str = "Iter " + str(i)
        log_str += " mult " + str(np.round(tmp_mult, 2))
        log_str += "\tE = " + str(np.round(result.total_energies[0], 7))
        log_str += "\tP = " + str(penalty)
        log_str += "\tE-P = " + str(np.round(result.total_energies[0] - penalty, 7))

        myLogger.info(log_str)

        if accectable_result:
            partial_results.append((result, penalty/tmp_mult, par))
        if accectable_result and penalty/tmp_mult < 1e-8 and i > 4:
            break


        if not accectable_result and i == iter_max - 1:
            partial_results.append((result, penalty/tmp_mult, par))

    result, optimal_par = find_best_result(partial_results)

    result = dummy_vqe(options, optimal_par)

    return result

def dummy_vqe(options, optimal_par):
    myLogger.info('inizio dummy_vqe')
    options['optimizer'] = COBYLA(maxiter=0)
    options['init_point'] = optimal_par

    new_result = solve_hamiltonian_vqe(options)

    myLogger.info('fine dummy_vqe')
    return new_result


def get_num_par(varform, mol_type):
    num_pars = 0

    if mol_type == 'H3+':
        if varform == 'TwoLocal':
            num_pars = 16
        elif varform == 'SO(4)':
            num_pars = 18
        elif varform == 'UCCSD':
            num_pars = 16
        elif varform == 'EfficientSU(2)':
            num_pars = 8
    elif mol_type == 'Na-':
        if varform == 'TwoLocal':
            num_pars = 24
        else:
            raise Exception('varform not yet implemented for this mol')
    elif 'H2O' in mol_type:
        if varform == 'TwoLocal':
            num_pars = 16
        elif varform == 'EfficientSU(2)':
            num_pars = 8
        elif varform == 'SO(4)':
            num_pars = 30
        elif varform == 'UCCSD':
            num_pars = 24
        else:
            raise Exception('varform not yet implemented for this mol')
    elif mol_type == 'C2H4':
        if varform == 'TwoLocal':
            num_pars = 8
        else:
            raise Exception('varform not yet implemented for this mol')
    elif 'H2' in mol_type:
        if varform == 'TwoLocal':
            num_pars = 8
        elif varform == 'SO(4)':
            num_pars = 6
        elif varform == 'UCCSD':
            num_pars = 8
        elif varform == 'EfficientSU(2)':
            num_pars = 4
    elif 'H4' in mol_type:
        if varform == 'TwoLocal':
            num_pars = 24
        elif varform == 'SO(4)':
            num_pars = 30
        elif varform == 'UCCSD':
            num_pars = 24
        elif varform == 'EfficientSU(2)':
            num_pars = 48
    elif 'Li2' in mol_type:
        if varform == 'TwoLocal':
            num_pars = 72
        else:
            raise Exception('varform not yet implemented for this mol')
    elif 'LiH' == mol_type:
        if varform == 'TwoLocal':
            num_pars = 16
        elif varform == 'EfficientSU(2)':
            num_pars = 8
        else:
            raise Exception('varform not yet implemented for this mol')
    else:
        raise Exception('mol_type not totally implemented')

    return num_pars

def solve_lag_aug_series_vqe(options):
    iter_max = options['series']['itermax']

    if 'init_point' not in options:
        par = np.random.rand(get_num_par(options['var_form_type'],
                                         options['molecule']['molecule']))
    else:
        par = options['init_point']

    mult = 0.01
    step = options['series']['step']

    lamb = options['series']['lamb']

    global PARAMETERS
    PARAMETERS = [par]
    for i in range(iter_max):
        tmp_mult = mult + step * i

    lag_op_list = []

    for single_op in options['lagrange']['operators']:
        operatore = (single_op[0],
                     single_op[1],
                     float(tmp_mult))
        lag_op_list.append(operatore)

    options['lagrange']['operators'] = lag_op_list

    options['init_point'] = par

    result = solve_aug_lagrangian_vqe(options, lamb)

    penalty = tmp_mult*((result.num_particles[0] - operatore[1])**2)
    penalty -= lamb * (result.num_particles[0] - operatore[1])

    log_str = "Iter " + str(i)
    log_str += " mult " + str(np.round(tmp_mult, 2))
    log_str += " lamb " + str(lamb)
    log_str += "\tE = " + str(np.round(result.total_energies[0], 7))
    log_str += "\tE-P = " + str(np.round(result.total_energies[0] - penalty, 7))
    myLogger.info(log_str)

    par = PARAMETERS[len(PARAMETERS) - 1]

    for operatore in lag_op_list:
        if operatore[0] == 'number':
            lamb = lamb - tmp_mult*2*(result.num_particles[0] - operatore[1])
        if operatore[0] == 'spin-squared':
            lamb = lamb - tmp_mult*2*(result.total_angular_momentum[0] - operatore[1])
        if operatore[0] == 'spin-z':
            lamb = lamb - tmp_mult*2*(result.magnetization[0] - operatore[1])

    #print(result.total_energies[0] - penalty, " ", penalty)
    return result

def solve_VQE(options):
    global PARAMETERS
    PARAMETERS = []

    global LAST_OPTIMIZER_OPT
    LAST_OPTIMIZER_OPT = None

    if not options['lagrange']['active']:
        vqe_result = solve_hamiltonian_vqe(options)
    elif not options['lagrange']['series']:
        lag_result = solve_lagrangian_vqe(options)
        optimal_par = PARAMETERS[len(PARAMETERS) - 1]
        vqe_result = dummy_vqe(options, optimal_par)
    else:
        if options['lagrange']['augmented']:
            vqe_result = solve_lag_aug_series_vqe(options)
        else:
            vqe_result = solve_lag_series_vqe(options)

    return vqe_result, PARAMETERS[len(PARAMETERS)-1]
