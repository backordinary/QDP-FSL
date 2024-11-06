# https://github.com/Oelfol/Dynamics/blob/ed8b55297aa488444e7cd12e1352e4f1b88b525f/HeisenbergCodes/HelpingFunctions.py
###########################################################################
# HelpingFunctions.py
# Part of HeisenbergCodes
# Updated January '21
#
# General accessory functions, including read/write.
###########################################################################

###########################################################################
# CONTENTS:
# get_pauli_matrices()
# init_spin_state(initialpsi, num_states)
# spin_op(operator, site, n, unity)
# gen_pairs(n, auto, open_chain)
# gen_m(leng, steps)
# commutes(a, b)
# sort_counts(count, qs, shots)
# sort_counts_no_div(count, qs)
# choose_control_gate(choice, qc, c, t)
# real_or_imag_measurement(qc, j)
# zz_operation(qc, a, b, delta)
# heis_pauli_circuit(a, b, num_spins, anc, op)
# post_rotations(num_spins, op, anc)
# three_cnot_evolution(qc, pair, ancilla, j, t, dt, trotter_steps, ising, a_constant)
# execute_real(qc, device, shots)
# run_circuit(anc, qc, noise, ibmq_params, n)
# gen_even_odd_pairs(n, open_chain)
# choose_RM(counts, num_qubits, RMfilename, RM=False)
# write_data(vec, loc)
# write_numpy_array(array, loc)
# read_numpy_array(filename)
# read_var_file(filename)
###########################################################################

from qiskit import execute
import numpy as np
import scipy.sparse as sps
import math
import csv
import IBMQSetup as setup
import ReadoutMitigation as IBU
from qiskit import QuantumCircuit

# ==================================== Virtual to hardware qubit mappings ============================================ >
# All examples except Joel use ibmq_ourense
# Initial layouts format: Virtual is index, physical is the number at the index

# Initial Layouts for Joel magnetization (casablanca) -> ancilla is always hardware 1
il1 = [1, 2, 3, 0, 4, 5, 6]
il2 = [1, 0, 2, 3, 4, 5, 6]
il3 = [1, 5, 0, 2, 3, 4, 6]
il4 = [1, 4, 6, 0, 2, 3, 5]
il5 = [1, 4, 5, 6, 0, 2, 3]
il6 = [1, 3, 4, 5, 6, 0, 2]
initial_layouts_joel = [il1, il2, il3, il4, il5, il6]
# To enable joel:
joel = True

# Initial Layouts for everything else (ourense) --> ancilla is always hardware 2:
initial_layouts_ee = [2, 1, 0, 3, 4]


# ========================================= Pauli matrices and pseudospin operators ================================== >

sx = sps.csc_matrix(np.array([[0, 1], [1, 0]]))
sy = sps.csc_matrix(np.complex(0, 1) * np.array([[0, -1], [1, 0]]))
sz = sps.csc_matrix(np.array([[1, 0], [0, -1]]))
identity = sps.csc_matrix(np.eye(2, dtype=complex))
plus = sps.csc_matrix(sx * (1 / 2) + np.complex(0, 1) * sy * (1 / 2))
minus = sps.csc_matrix(sx * (1 / 2) - np.complex(0, 1) * sy * (1 / 2))


def get_pauli_matrices():
    return sx, sy, sz, identity, plus, minus

# ======================================= Helper functions for classical calculations ================================ >


def init_spin_state(initialpsi, num_states):
    # initial psi is the computational-basis index
    psi_0 = np.zeros([num_states, 1], complex)
    psi_0[initialpsi, 0] = 1
    return sps.csc_matrix(psi_0)


def spin_op(operator, site, n, unity):
    """
    Generates Pauli spin operators & spin-plus/-minus operators
    :param operator: operator {'x','y','z','+','-'} (str)
    :param site: site index of operator (int)
    :param n: number of sites (int)
    :param unity: sets h-bar/2 to unity (bool)
    """
    array = identity
    ops_list = ['x', 'y', 'z', '+', '-']
    pauli_ops = [(1 / 2) * sx, (1 / 2) * sy, (1 / 2) * sz, plus, minus]

    if unity:
        # set h-bar / 2 == 1 instead of h-bar == 1
        pauli_ops = [sx, sy, sz, 2 * plus, 2 * minus]

    if site == (n - 1):
        array = pauli_ops[ops_list.index(operator)]
    for x in reversed(range(0, n - 1)):
        if x == site:
            array = sps.kron(array, pauli_ops[ops_list.index(operator)])
        else:
            array = sps.kron(array, identity)

    return array


def gen_pairs(n, auto, open_chain):
    """
    auto: whether needing autocorrelations, not for hamiltonian matrix
    open_chain: includes periodic boundary conditions of False
    """
    nn, autos = [], []
    for p in range(n - 1):
        if auto:
            autos.append((p, p))
        nn.append((p, p + 1))
    if auto:
        autos.append((n - 1, n - 1))
    if n > 2 and not open_chain:
        nn.append((0, n - 1))
    return nn, autos


def gen_m(leng, steps):
    # Generate an empty data matrix
    return sps.lil_matrix(np.zeros([leng, steps]), dtype=complex)


def commutes(a, b):
    # Test whether operators commute
    comp = a.dot(b) - b.dot(a)
    comp = comp.toarray().tolist()
    if np.count_nonzero(comp) == 0:
        return True
    else:
        return False


# ============================================== Helper functions for Qiskit Simulations ============================= >


def sort_counts(count, qs, shots):
    # sort counts and divide out shots
    vec = []
    for i in range(2 ** qs):
        binary = np.binary_repr(i).zfill(qs)
        if binary in count.keys():
            vec.append(count[binary] / shots)
        else:
            vec.append(0.0)
    return vec


def sort_counts_no_div(count, qs):
    # sort counts without dividing out shots
    vec = []
    for i in range(2 ** qs):
        binary = np.binary_repr(i).zfill(qs)
        if binary in count.keys():
            vec.append(count[binary])
        else:
            vec.append(0.0)
    return vec


def choose_control_gate(choice, qc, c, t):
    # Applying chosen controlled unitary for correlations and magnetization
    if choice == 'x':
        qc.cx(control_qubit=c, target_qubit=t)
    elif choice == 'y':
        qc.cy(control_qubit=c, target_qubit=t)
    elif choice == 'z':
        qc.cz(control_qubit=c, target_qubit=t)


def real_or_imag_measurement(qc, j):
    # For ancilla-assisted measurements
    if j == 0:
        qc.h(0)
    elif j == 1:
        qc.rx(-1 * math.pi / 2, 0)


def zz_operation(qc, a, b, delta):
    # For time evolution operator
    qc.cx(a, b)
    qc.rz(2 * delta, b)
    qc.cx(a, b)


def heis_pauli_circuit(a, b, num_spins, op):
    # For vqe, obtaining hamiltonian pauli string circuits as controlled gates
    c, q = 1, num_spins + 1
    qc = QuantumCircuit(q, c)
    if op == 'x':
        qc.cx(0, a + 1)
        qc.cx(0, b + 1)
    elif op == 'y':
        qc.cy(0, a + 1)
        qc.cy(0, b + 1)
    elif op == 'z':
        qc.cz(0, a + 1)
        qc.cz(0, b + 1)

    # for magnetic field terms: (param b is None)
    elif op == 'z*':
        qc.cz(0, a + 1)

    return qc


def three_cnot_evolution(qc, pair, ancilla, j, t, dt, trotter_steps, ising, a_constant):
    a_, b_ = pair[0] + ancilla, pair[1] + ancilla
    if ising:
        zz_operation(qc, a_, b_, j * t * dt  / trotter_steps)
    else:
        delta = j * t * dt / trotter_steps
        qc.cx(a_, b_)
        qc.rx(2 * delta - math.pi / 2, a_)
        qc.rz(2 * delta * a_constant, b_)
        qc.h(a_)
        qc.cx(a_, b_)
        qc.h(a_)
        qc.rz(- 2 * delta, b_)
        qc.cx(a_, b_)
        qc.rx(math.pi / 2, a_)
        qc.rx(-math.pi / 2, b_)


def grouped_three_cnot_evolution(qc, pairs, ancilla, j, t, dt, trotter_steps, ising, a_constant):
    # time evolution for isotropic heisenberg and commuting set of pairs even/odd
    delta = j * t * dt / trotter_steps
    fixed_pairs = [(pairs[i][0] + ancilla, pairs[i][1] + ancilla) for i in range(len(pairs))]
    for pair1 in fixed_pairs:
        qc.cx(pair1[0], pair1[1])
    qc.barrier()
    for pair2 in fixed_pairs:
        qc.rx(2 * delta - math.pi / 2, pair2[0])
        qc.rz(2 * delta * a_constant, pair2[1])
        qc.h(pair2[0])
    qc.barrier()
    for pair3 in fixed_pairs:
        qc.cx(pair3[0], pair3[1])
        qc.h(pair3[0])
        qc.rz(- 2 * delta, pair3[1])
    qc.barrier()
    for pair4 in fixed_pairs:
        qc.cx(pair4[0], pair4[1])
        qc.rx(math.pi / 2, pair4[0])
        qc.rx(-math.pi / 2, pair4[1])


def execute_real(qc, device, shots):
    # run circuit on real hardware
    result = execute(qc, backend=device, shots=shots).result()
    return result


def choose_RM(counts, num_qubits, RMfilename, RM=False):
    # choose whether to use readout mitigation ---> will be automatic here, but this is old code
    # counts: counts directly after running the circuit
    # num_qubits: number of qubits measured

    if RM:
        probs = sort_counts_no_div(counts[0], num_qubits)
        probs = IBU.unfold(RMfilename, setup.shots, probs, num_qubits)
        return probs
    else: # this doesnt get used anymore
        probs = sort_counts(counts[0], num_qubits, setup.shots)
        return probs


def run_circuit(anc, qc, noise, ibmq_params, n, site, rm_filename):
    [device, nm, bg, simulator, real_sim, coupling_map] = ibmq_params
    # real_sim: if True, run on the real backend, otherwise noise model.
    shots = setup.shots

    # deal w/ layouts
    initial_layout = []
    if joel:
        for l in initial_layouts_joel[0]: #[site]: # TODO temporary until can do more readout arrays
            initial_layout.append(l)
    else:
        for l in initial_layouts_ee: 
            initial_layout.append(l)

    # remove items in initial_layout which we do not need.
    init_layout = initial_layout[:n + 1]

    num_qubits_measured = 0

    if anc == 1:
        qc.measure(0, 0)
        num_qubits_measured += 1
    else:
        for x in range(n):
            qc.measure(x, x)
        num_qubits_measured += n

    counts = []

    # Ideal simulation :
    if not noise: #coupling_map=coupling_map #optimization_level=3
        result = execute(qc, backend=simulator, shots=shots).result()
        counts += [result.get_counts(i) for i in range(len(result.results))]

    # Noise model or hardware:
    else:
        if real_sim:
            result = execute_real(qc, device, shots)
            counts += [result.get_counts(i) for i in range(len(result.results))]
        else:
            result = execute(qc, backend=simulator, shots=shots, noise_model=nm, basis_gates=bg,
                         optimization_level=0, initial_layout=init_layout).result()
            counts += [result.get_counts(i) for i in range(len(result.results))]

    if anc == 1:
        probs = choose_RM(counts, num_qubits_measured, rm_filename, False) #True) # TODO fix so this is optional
        return probs[0] - probs[1]
    
    # occupation probabilities will not work anymore, TODO 
    #else:
    #    measurements = sort_counts(counts[0], n, shots)
    #    return measurements


def gen_even_odd_pairs(n, open_chain):
    # Used only for second-order trotter
    nn_even = []
    nn_odd = []
    for p in range(n - 1):
        if p%2 == 0:
            nn_even.append((p, p + 1))
        else:
            nn_odd.append((p, p + 1))
    if n > 2 and not open_chain:
        if (n - 1) % 2 == 0:
            nn_even.append((0, n - 1))
        else:
            nn_odd.append((0, n - 1))
    return [nn_even, nn_odd]


def gen_num_bonds(n, open):
    # given a Heisenberg chain length n and choice of open or closed, find number of bonds
    # for even/odd pair splitting in trotter  -- assumes only next-nearest
    no_bonds = 0
    if open:
        no_bonds += n - 1
    else:
        no_bonds += n
    return no_bonds


# =============================== Writing / Reading Helpers ======================================================= >


def write_data(vec, loc):
    # write data from a 1d list into a csv file by rows
    with open(loc,'a',newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(vec)
    csvFile.close()


def write_numpy_array(array, loc):
    # write a numpy array into a text file
    np.savetxt(loc, array, delimiter=",")


def read_numpy_array(filename):
    # read numpy array from a textfile
    lines = np.loadtxt(filename, delimiter=",", unpack=False)
    return lines


def read_var_file(filename):
    data = []
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
        data += list(lines)
    csvfile.close()
    return data
