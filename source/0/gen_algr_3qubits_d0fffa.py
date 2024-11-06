# https://github.com/Viperr96/Qunatum_things/blob/c52e8e066c7f99b47ec3cec942d58528bf81c6b9/gen/gen_algr_3qubits.py
# import matplotlib.pyplot as plt
import numpy as np
import getpass, time
from math import pi
from h5py import File
from argparse import ArgumentParser
import os, os.path

from qiskit import IBMQ, QuantumCircuit, ClassicalRegister, QuantumRegister, execute, Aer, BasicAer
from qiskit.providers.aer import noise
from qiskit.tools.monitor import job_monitor, backend_monitor, backend_overview
from qiskit.quantum_info.analyzation.average import average_data
from qiskit.providers.ibmq import least_busy
from qiskit.providers.exceptions import JobError, JobTimeoutError



from qiskit.tools.visualization import plot_histogram, circuit_drawer

def do_job_on_simulator(real_backend , circuits:  list):

    gate_times = [
        ('u1', None, 0), ('u2', None, 100), ('u3', None, 200),
        ('cx', [1, 0], 678), ('cx', [1, 2], 547), ('cx', [2, 3], 721),
        ('cx', [4, 3], 733), ('cx', [4, 10], 721), ('cx', [5, 4], 800),
        ('cx', [5, 6], 800), ('cx', [5, 9], 895), ('cx', [6, 8], 895),
        ('cx', [7, 8], 640), ('cx', [9, 8], 895), ('cx', [9, 10], 800),
        ('cx', [11, 10], 721), ('cx', [11, 3], 634), ('cx', [12, 2], 773),
        ('cx', [13, 1], 2286), ('cx', [13, 12], 1504), ('cx', [], 800)
    ]
    properties = real_backend.properties()
    coupling_map = real_backend.configuration().coupling_map
    noise_model = noise.device.basic_device_noise_model(properties, gate_times=gate_times)
    basis_gates = noise_model.basis_gates
    simulator = Aer.get_backend('qasm_simulator')
    job = execute(circuits,
                  simulator,
                  noise_model=noise_model,
                  coupling_map=coupling_map,
                  basis_gates=basis_gates,
                  memory = True)
    return job

def make_rotation(qubit_quantity: int, circuit: QuantumCircuit, registers: QuantumRegister, theta, phi, lada):

    """



    Rotates and entangle qubits in the circuit
.





    :param circuit: a QuantumCircuit object comprising two qubits



    :param registers: list of registers involved in the circuit



    :param angles: list of tuples (theta, lambda, phi) with rotation angles



    :return:



    """

    if qubit_quantity == 2:

        circuit.u3(theta[0], phi[0], lada[0], registers[0])

        circuit.u3(theta[1], phi[1], lada[1], registers[1])

        circuit.cx(registers[1], registers[0])

    if qubit_quantity == 3:

        circuit.u3(theta[0], phi[0], lada[0], registers[0])

        circuit.u3(theta[1], phi[1], lada[1], registers[1])

        circuit.u3(theta[2], phi[2], lada[2], registers[2])

        circuit.cx(registers[1], registers[0])

        circuit.cx(registers[2], registers[1])

        circuit.cx(registers[0], registers[2])

    if qubit_quantity == 4:

        circuit.u3(theta[0], phi[0], lada[0], registers[0])

        circuit.u3(theta[1], phi[1], lada[1], registers[1])

        circuit.u3(theta[2], phi[2], lada[2], registers[2])

        circuit.u3(theta[3], phi[3], lada[3], registers[3])

        circuit.cx(registers[1], registers[0])

        circuit.cx(registers[2], registers[1])

        circuit.cx(registers[3], registers[2])

        circuit.cx(registers[0], registers[3])


def get_energy(qubit_quantity: int, exchange, bj, result, circuits: list) -> float:


    """

    Calculate Heisenberg energy

    :param qubit_quantity: qubit quantity

    :param exchange: exchange constant

    :param result: a job object storing the results of quantum calculation

    :param bj: B/J

    :param circuits: list containing XX YY and ZZ circuits

    :return:

    """

    res = 0

    if qubit_quantity == 2:
        codes = {'00': 1, '01': -1, '10': -1, '11': 1}
        res = exchange/4 * (average_data(result.get_counts(circuits[0]), codes) +
                            average_data(result.get_counts(circuits[1]), codes) +
                            average_data(result.get_counts(circuits[2]), codes)) + \
              bj * exchange * get_magnetization(qubit_quantity, result, [circuits[2]])

    if qubit_quantity == 3:

        first_correlated = {'000': 1, '100': 1, '011': 1, '111': 1, '001': -1, '010': -1, '101': -1, '110': -1}
        second_correlated = {'000': 1, '001': 1, '110': 1, '111': 1, '100': -1, '010': -1, '101': -1, '011': -1}

        # smart_correlated = {
        #     '12', {'000': 1, '100': 1, '011': 1, '111': 1, '001': -1, '010': -1, '101': -1, '110': -1, },
        #     '23', {'000': 1, '001': 1, '110': 1, '111': 1, '100': -1, '010': -1, '101': -1, '011': -1, }
        # }

        res = exchange/4 * (average_data(result.get_counts(circuits[0]), first_correlated) +
                            average_data(result.get_counts(circuits[1]), first_correlated) +
                            average_data(result.get_counts(circuits[2]), first_correlated) +
                            average_data(result.get_counts(circuits[0]), second_correlated) +
                            average_data(result.get_counts(circuits[1]), second_correlated) +
                            average_data(result.get_counts(circuits[2]), second_correlated)) + \
              bj * exchange * get_magnetization(qubit_quantity, result, [circuits[2]])

    if qubit_quantity == 4:

        first_correlated = {'0000': 1, '1100': 1, '1000': 1, '0100': 1, '0011': 1, '1111': 1, '1011': 1, '0111': 1,
                            '0001': -1, '1101': -1, '1001': -1, '0101': -1, '0010': -1, '1110': -1, '1010': -1, '0110': -1} #**11
        second_correlated = {'0000': 1, '1001': 1, '1000': 1, '0001': 1, '0110': 1, '1111': 1, '1110': 1, '0111': 1,
                             '0100': -1, '1101': -1, '1100': -1, '0101': -1, '0010': -1, '1011': -1, '1010': -1, '0011': -1,}#*11*
        third_correlated = {'0000': 1, '0011': 1, '0010': 1, '0001': 1, '1100': 1, '1111': 1, '1110': 1, '1101': 1,
                             '1000': -1, '1011': -1, '1010': -1, '1001': -1, '0100': -1, '0111': -1, '0110': -1, '0101': -1,}#11**

        res = exchange/4 * (average_data(result.get_counts(circuits[0]), first_correlated) +
                            average_data(result.get_counts(circuits[1]), first_correlated) +
                            average_data(result.get_counts(circuits[2]), first_correlated) +
                            average_data(result.get_counts(circuits[0]), second_correlated) +
                            average_data(result.get_counts(circuits[1]), second_correlated) +
                            average_data(result.get_counts(circuits[2]), second_correlated) +
                            average_data(result.get_counts(circuits[0]), third_correlated) +
                            average_data(result.get_counts(circuits[1]), third_correlated) +
                            average_data(result.get_counts(circuits[2]), third_correlated)) + \
              bj * exchange * get_magnetization(qubit_quantity, result, [circuits[2]])

    return res




def get_fitness(result, circuits: list, codes1: dict, codes2: dict, codes: dict) -> float:
    """
    Calculate fitness

    :param exchange: exchange constant
    :param result: a job object storing the results of quantum calculation
    :param circuits: list containing XX YY and ZZ circuits
    :param codes:
    :return:
    """
    res =  abs(average_data(result.get_counts(circuits[0]), codes1))/2\
           +abs(average_data(result.get_counts(circuits[0]), codes2))/2\
           +abs(average_data(result.get_counts(circuits[1]), codes1))/2\
           +abs(average_data(result.get_counts(circuits[1]), codes2))/2\
           +abs(average_data(result.get_counts(circuits[2]), codes1))/2\
           +abs(average_data(result.get_counts(circuits[2]), codes2))/2\
           +abs(average_data(result.get_counts(circuits[0]), codes)+1)\
           +abs(average_data(result.get_counts(circuits[1]), codes)+1)\
           +abs(average_data(result.get_counts(circuits[2]), codes)+1)\
           +abs(average_data(result.get_counts(circuits[3]), codes))\
           +abs(average_data(result.get_counts(circuits[4]), codes))\
           +abs(average_data(result.get_counts(circuits[5]), codes))\
           +abs(average_data(result.get_counts(circuits[6]), codes))\
           +abs(average_data(result.get_counts(circuits[7]), codes))\
           +abs(average_data(result.get_counts(circuits[8]), codes))

    return res

def get_magnetization(qubit_quantity: int, result, circuits: list) -> float:

    res = 0

    if qubit_quantity == 2:

        observable_first = {'00': 1, '01': -1, '10': 1, '11': -1}

        observable_second = {'00': 1, '01': 1, '10': -1, '11': -1}

        res = (average_data(result.get_counts(circuits[0]), observable_first) + average_data(result.get_counts(circuits[0]), observable_second)) / 2

    if qubit_quantity == 3:

        observable_first = {'000': 1, '100': 1, '011': -1, '111': -1, '001': -1, '010': 1, '101': -1, '110': 1, }

        observable_second = {'000': 1, '100': 1, '011': -1, '111': -1, '001': 1, '010': -1, '101': 1, '110': -1, }

        observable_third = {'000': 1, '100': -1, '011': 1, '111': -1, '001': 1, '010': 1, '101': -1, '110': -1, }

        res = (average_data(result.get_counts(circuits[0]), observable_first) +
               average_data(result.get_counts(circuits[0]), observable_second) +
               average_data(result.get_counts(circuits[0]), observable_third)) / 2

    if qubit_quantity == 4:

        observable_first = {'0000': 1, '1110': 1, '1000': 1, '0100': 1, '0010': 1, '1100': 1, '0110': 1, '1010': 1,
                            '0001': -1, '1111': -1, '1001': -1, '0101': -1, '0011': -1, '1101': -1, '0111': -1, '1011': -1} #***1

        observable_second = {'0000': 1, '1101': 1, '1000': 1, '0100': 1, '0001': 1, '1100': 1, '0101': 1, '1001': 1,
                            '0010': -1, '1111': -1, '1010': -1, '0110': -1, '0011': -1, '1110': -1, '0111': -1, '1011': -1} #**1*

        observable_third = {'0000': 1, '1011': 1, '1000': 1, '0010': 1, '0001': 1, '1010': 1, '0011': 1, '1001': 1,
                            '0100': -1, '1111': -1, '1100': -1, '0110': -1, '0101': -1, '1110': -1, '0111': -1, '1101': -1} #*1**

        observable_fourth = {'0000': 1, '0111': 1, '0100': 1, '0010': 1, '0001': 1, '0110': 1, '0011': 1, '0101': 1,
                            '1000': -1, '1111': -1, '1100': -1, '1010': -1, '1001': -1, '1110': -1, '1011': -1, '1101': -1} #1***

        res = (average_data(result.get_counts(circuits[0]), observable_first) +
               average_data(result.get_counts(circuits[0]), observable_second) +
               average_data(result.get_counts(circuits[0]), observable_third) +
               average_data(result.get_counts(circuits[0]), observable_fourth)) / 2

    return res

def flush_angles(theta, lada, phi, prefix=''):



    """


    Write current angles to hdf5 file.

    :param theta: theta rotation angle


    :param lada: lambda rotation angle


    :param phi: phi rotation angle


    :param prefix: prefix for output file


    :return: Nothing


    """


    with File(prefix + 'angles.h5', 'w') as f:


        f['/angles/theta'] = theta


        f['/angles/labmda'] = lada


        f['/angles/phi'] = phi


def read_angles(fn, num_genome=None):

    """

    Read angles from an hdf5 file.

    :param fn: path to hdf5 file


    :param num_genome: number of noble genomes in current caluclation fo check (optional)


    :return: tuple of noble angles


    """

    with File(fn, 'r') as f:


        theta = f['/angles/theta'][:]


        lada = f['/angles/labmda'][:]


        phi = f['/angles/phi'][:]





    if num_genome is not None:


        if theta.shape[0] != num_genome or lada.shape[0] != num_genome or phi.shape[0] != num_genome:


            raise ValueError('Wrong shape of input data! Check the number of genomes!')





    return (theta, lada, phi)


def select_genomes(qubit_quantity: int, energy: list, num_noble_genome: int, theta: np.array, lada: np.array, phi: np.array) -> tuple:


    """

    Select noble genomes corresponding to minimum energies.

    :param energy: list of energies from calculation


    :param num_genome: number of noble genomes in current caluclation


    :param theta: theta rotation angles


    :param lada: lambda rotation angles


    :param phi: phi rotation angles


    :return: list of noble angles


    """

    depth_loc = theta.shape[1]


    noble_theta = np.ndarray((num_noble_genome, depth_loc, qubit_quantity), dtype=np.float32)


    noble_lada = np.ndarray((num_noble_genome, depth_loc, qubit_quantity), dtype=np.float32)


    noble_phi = np.ndarray((num_noble_genome, depth_loc, qubit_quantity), dtype=np.float32)


    min_vals = energy.copy()

    min_vals.sort()
    print(min_vals)
    # min_vals = min_vals[::-1]

    min_vals = min_vals[:num_noble_genome]
    print(min_vals)



    idx = 0

    numbers=[]
    ideal_angle_n = 0

    for n, e in enumerate(energy):
        if e in min_vals:




            noble_theta[idx,:,:] = theta[n,:,:]

            noble_lada[idx,:,:] = lada[n,:,:]

            noble_phi[idx,:,:] = phi[n,:,:]

            numbers.append(n)

            if e == min_vals[0]:
                ideal_angle_n = n

            idx += 1

            if idx == num_noble_genome:

                break


    print('igenome - fitness')
    for idx in numbers:
        print(idx, energy[idx])
    return (noble_theta, noble_lada, noble_phi, ideal_angle_n)


def evolve_genomes(qubit_quantity: int, noble_theta, noble_lada, noble_phi, angle_variation):

    # theta = np.random.random_sample((num_genome, depth, 2)) * pi
    # phi = np.random.random_sample((num_genome, depth, 2)) * 2 * pi
    # lada = np.random.random_sample((num_genome, depth, 2)) * 2 * pi

    for igenome in range(num_noble_genome):

        theta[igenome,:,:]=noble_theta[igenome,:,:]
        phi[igenome,:,:]=noble_phi[igenome,:,:]
        lada[igenome,:,:]=noble_lada[igenome,:,:]

    ran_theta=np.random.randint(3,size=(num_genome,depth,qubit_quantity))
    ran_phi=np.random.randint(3,size=(num_genome,depth,qubit_quantity))
    ran_lada=np.random.randint(3,size=(num_genome,depth,qubit_quantity))

    for igenome in range(num_noble_genome,2*num_noble_genome):

        theta[igenome,:,:]= ( 1-( 1-ran_theta[igenome-num_noble_genome,:,:] )*angle_variation ) * noble_theta[igenome-num_noble_genome,:,:]
        phi[igenome,:,:]= (1-(1-ran_phi[igenome-num_noble_genome,:,:])*angle_variation) * noble_phi[igenome-num_noble_genome,:,:]
        lada[igenome,:,:]= (1-(1-ran_lada[igenome-num_noble_genome,:,:])*angle_variation) * noble_lada[igenome-num_noble_genome,:,:]

    ran_theta=np.random.randint(2,size=(num_genome,depth,qubit_quantity))
    ran_phi=np.random.randint(2,size=(num_genome,depth,qubit_quantity))
    ran_lada=np.random.randint(2,size=(num_genome,depth,qubit_quantity))

    for igenome in range(2*num_noble_genome,3*num_noble_genome):

        theta[igenome,:,:]= ran_theta[igenome-2*num_noble_genome,:,:]*theta[igenome-2*num_noble_genome,:,:]+(1-ran_theta[igenome,:,:])*theta[igenome-2*num_noble_genome+1,:,:]
        phi[igenome,:,:]=ran_phi[igenome-2*num_noble_genome,:,:]*phi[igenome-2*num_noble_genome,:,:]+(1-ran_phi[igenome,:,:])*phi[igenome-2*num_noble_genome+1,:,:]
        lada[igenome,:,:]=ran_lada[igenome-2*num_noble_genome,:,:]*lada[igenome-2*num_noble_genome,:,:]+(1-ran_lada[igenome,:,:])*lada[igenome-2*num_noble_genome+1,:,:]

    ran_theta=np.random.randint(2,size=(num_genome,depth,qubit_quantity))
    ran_phi=np.random.randint(2,size=(num_genome,depth,qubit_quantity))
    ran_lada=np.random.randint(2,size=(num_genome,depth,qubit_quantity))

    for igenome in range(3*num_noble_genome,4*num_noble_genome):

        theta[igenome,:,:]=ran_theta[igenome-3*num_noble_genome+1,:,:]*theta[igenome-3*num_noble_genome,:,:]+(1-ran_theta[igenome,:,:])*theta[igenome-3*num_noble_genome+2,:,:]
        phi[igenome,:,:]=ran_phi[igenome-3*num_noble_genome+1,:,:]*phi[igenome-3*num_noble_genome,:,:]+(1-ran_phi[igenome,:,:])*phi[igenome-3*num_noble_genome+2,:,:]
        lada[igenome,:,:]=ran_lada[igenome-3*num_noble_genome+1,:,:]*lada[igenome-3*num_noble_genome,:,:]+(1-ran_lada[igenome,:,:])*lada[igenome-3*num_noble_genome+2,:,:]


    for igenome in range(4*num_noble_genome,5*num_noble_genome):

        theta[igenome,:,:] = np.random.random_sample(qubit_quantity) * pi
        phi[igenome,:,:] = np.random.random_sample(qubit_quantity) * 2 * pi
        lada[igenome,:,:] = np.random.random_sample(qubit_quantity) * 2 * pi

    return [theta, lada, phi]


p = ArgumentParser()



p.add_argument('--prefix', default='', help='Prefix for all output files.')



p.add_argument('--load-angles', action='store_true', help='Save angles during calculation.')


p = p.parse_args()


if p.prefix:

 p.prefix += '_'


provider = IBMQ.enable_account('ed1f7070919a8ce0469e69c1cb5b5dc1e114879caada8d0ce25d6e28e91b40c90209146d85eec5118e2584667b02e9662802f0ffaa2494c72375a4906e129fdf')
print('Account loaded')

# name_backend='ibmq_16_melbourne'
#name_backend='ibmqx4'

# backend = least_busy(IBMQ.backends(filters=lambda x: not x.configuration().simulator))
backend = provider.get_backend('ibmq_16_melbourne')
backend_monitor(backend)

iteration = 2
num_qubits = 4
Jexch=1
device_shots = 4096
depth=1
num_genome = 100
num_noble_genome = 20
num_iter = 25
angle_mutation = 0.1
working_backend = 0



theta=np.random.random_sample((num_genome,depth,num_qubits))*pi
phi=np.random.random_sample((num_genome,depth,num_qubits))*2*pi
lada=np.random.random_sample((num_genome,depth,num_qubits))*2*pi

backend_dict = {0: 'simulator', 1: 'simualtor_with_noise', 2: 'real_device'}
folder_name =  backend_dict[working_backend] + '-' + str(backend) + '-' + str(iteration) + "-random_start-num_qubits=" + str(num_qubits) + ",Jexch=" + str(Jexch) + \
              ",device_shots=" + str(device_shots) + ",depth=" + str(depth) + ",num_genome=" + str(num_genome) + \
              ",num_iter=" + str(num_iter) + ",angle_mutation=" + str(angle_mutation)

if not os.path.exists(folder_name):
    try:
        os.mkdir(folder_name)
    except OSError:
        print("Creation of the directory failed")


folder_name += "/"
if p.load_angles:

    noble_angles = read_angles(folder_name + 'angles.h5', num_noble_genome)

    print(noble_angles)

    theta, lada, phi = evolve_genomes(num_qubits, *noble_angles, angle_mutation)





result_path = folder_name + 'final_result.dat'
result_days = open(result_path, 'w')

# backend = least_busy(IBMQ.backends(filters=lambda x: not x.configuration().simulator))
# backend = IBMQ.get_backend('ibmq_16_melbourne')
# backend_monitor(backend)

for bj_step in range(0, 61):

    theta = np.random.random_sample((num_genome, depth, num_qubits)) * pi
    phi = np.random.random_sample((num_genome, depth, num_qubits)) * 2 * pi
    lada = np.random.random_sample((num_genome, depth, num_qubits)) * 2 * pi

    pres = 0.001
    prev_energy = 10
    bj_step = bj_step / 10
    energy_path = folder_name + str(bj_step) + '-energy.dat'
    energy_days = open(energy_path, 'w')
    correlator_path = folder_name + str(bj_step) + '-Z_correlator.dat'
    correlator_days = open(correlator_path, 'w')



    for i in range(num_iter):

        print('BJ ' + str(bj_step) + ', iteration numer ' + str(i))
        circuits = []

        energy_Heis = np.zeros((num_genome), np.float32)
        fitness = np.zeros((num_genome), np.float32)
        magnetization = np.zeros((num_genome), np.float32)
        corXX = np.zeros((num_genome), np.float32)
        corYY = np.zeros((num_genome), np.float32)
        corZZ = np.zeros((num_genome), np.float32)


        for igenome in range(num_genome):

            # Creating registers
            q = QuantumRegister(num_qubits)
            c = ClassicalRegister(num_qubits)
            # quantum circuit to make an entangled bell state
            singlet = QuantumCircuit(q, c)

            for idepth in range(depth):

                # lada[igenome,idepth,0] = 0
                # lada[igenome, idepth, 1] = 0
                # lada[igenome, idepth, 2] = 0

                make_rotation(num_qubits, singlet, q, theta[igenome,idepth,:],phi[igenome,idepth,:],lada[igenome,idepth,:])


            measureZZ = QuantumCircuit(q, c)
            measureYY = QuantumCircuit(q, c)
            measureXX = QuantumCircuit(q, c)

            for iqubit in range(num_qubits):

                measureZZ.measure(q[iqubit], c[iqubit])
                measureYY.sdg(q[iqubit])
                measureYY.h(q[iqubit])
                measureYY.measure(q[iqubit], c[iqubit])
                measureXX.h(q[iqubit])
                measureXX.measure(q[iqubit], c[iqubit])


            exec('singletZZ_%d = singlet+measureZZ' % igenome)
            exec('singletYY_%d = singlet+measureYY' % igenome)
            exec('singletXX_%d = singlet+measureXX' % igenome)

            exec('circuits.extend([singletXX_%d,singletYY_%d,singletZZ_%d])' % (igenome, igenome, igenome))

        result = 0
        no_error = True
        while no_error:
            try:
                if working_backend == 0:
                    job = execute(circuits, BasicAer.get_backend('qasm_simulator'), shots=device_shots, memory = True)
                if working_backend == 1:
                    job = do_job_on_simulator(backend, circuits)
                if working_backend == 2:
                    job = execute(circuits, backend=backend, shots=device_shots, memory = True)
                job_monitor(job)
                result = job.result()
                no_error = False
            except JobError:
                print(JobError)
                no_error = True

        observable_first = {'00': 1, '01': -1, '10': 1, '11': -1}
        observable_second = {'00': 1, '01': 1, '10': -1, '11': -1}
        observable_correlated = {'00': 1, '01': -1, '10': -1, '11': 1}

        for igenome in range(num_genome):

            exec('energy_Heis[igenome] = get_energy(num_qubits, Jexch, bj_step, result, [singletXX_%d, singletYY_%d, singletZZ_%d])' % (igenome, igenome, igenome))
            fitness[igenome] = 0
            exec('magnetization[igenome] = get_magnetization(num_qubits, result,[singletZZ_%d])' % igenome)
            exec('corXX[igenome] = average_data(result.get_counts(singletXX_%d), observable_correlated)' % igenome)
            exec('corYY[igenome] = average_data(result.get_counts(singletYY_%d), observable_correlated)' % igenome)
            exec('corZZ[igenome] = average_data(result.get_counts(singletZZ_%d), observable_correlated)' % igenome)

            # print('average energy_Heis=',energy_Heis[igenome])
            # print('average fitness=',fitness[igenome])
            # print('average magnetization=', magnetization[igenome])

        energy_for_file = energy_Heis.copy()
        energy_for_file.sort()
        energy_days.write(str(i))
        for i_energy in energy_for_file:
            energy_days.write(' ' + str(i_energy))
        energy_days.write('\n')
        energy_days.flush()

        nobles = select_genomes(num_qubits, energy_Heis, num_noble_genome, theta, lada, phi)
        ideal_angle_number = nobles[3]
        nobles = nobles[:3]

        correlator_days.write(str(i) + ' ' + str(energy_Heis[ideal_angle_number]) + ' ' + str(magnetization[ideal_angle_number]) + ' ' + str(fitness[ideal_angle_number]) + ' ' +
                              str(corXX[ideal_angle_number]) + ' ' + str(corYY[ideal_angle_number]) + ' ' + str(corZZ[ideal_angle_number]) + ' angles:')
        for pair in theta[ideal_angle_number, :, :]:
            correlator_days.write('{0} {1} '.format(*pair))

        for pair in phi[ideal_angle_number, :, :]:
            correlator_days.write('{0} {1} '.format(*pair))

        for pair in lada[ideal_angle_number, :, :]:
            correlator_days.write('{0} {1} '.format(*pair))
        correlator_days.write('\n')
        correlator_days.flush()

        if (abs(prev_energy - energy_Heis[ideal_angle_number]) < pres) or (i == num_iter - 1):

            memory_path = folder_name + str(bj_step) + '-memory.txt'
            memory_days = open(memory_path, 'w')
            memory_list = []
            exec('memory_list = result.get_memory(singletZZ_%d)' % ideal_angle_number)
            for item_list in memory_list:
                item_list = list(item_list)
                for item in item_list:
                    memory_days.write('%s ' %item)
                memory_days.write('\n')
            memory_days.flush()
            memory_days.close()



            result_days.write(str(bj_step) + ' ' + str(energy_Heis[ideal_angle_number]) + ' ' + str(magnetization[ideal_angle_number]) + ' ' + str(backend) + ' angles:')
            for pair in theta[ideal_angle_number,:,:]:
                result_days.write('{0} {1} '.format(*pair))

            for pair in phi[ideal_angle_number,:,:]:
                result_days.write('{0} {1} '.format(*pair))

            for pair in lada[ideal_angle_number,:,:]:
                result_days.write('{0} {1}'.format(*pair))

            result_days.write('\n')
            result_days.flush()
            break

        else:
            prev_energy = energy_Heis[ideal_angle_number]

        # if i == num_iter - 1:
        #     result_days.write(str(bj_step) + ' ' + str(energy_Heis[ideal_angle_number]) + ' ' + str(magnetization[ideal_angle_number]) +  ' ' + str(backend) + ' angles:')
        #     for pair in theta[ideal_angle_number,:,:]:
        #         result_days.write('{0} {1} '.format(*pair))
        #
        #     for pair in phi[ideal_angle_number,:,:]:
        #         result_days.write('{0} {1} '.format(*pair))
        #
        #     for pair in lada[ideal_angle_number,:,:]:
        #         result_days.write('{0} {1}'.format(*pair))
        #
        #     result_days.write('\n')
        #     result_days.flush()

        flush_angles(*nobles, folder_name)

        [theta, lada, phi] = evolve_genomes(num_qubits, nobles[0], nobles[1], nobles[2], angle_mutation)


    energy_days.close()
    correlator_days.close()

result_days.close()
print('end of VQE loop')



