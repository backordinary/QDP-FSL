# https://github.com/quantum-mining/quantum-mining-publication/blob/606bda884426474a12bccac69d0aad55804207ef/mine_solver/utils.py
import numpy as np
import copy
import sys
from qiskit import QuantumRegister, QuantumCircuit, BasicAer
from qiskit.quantum_info import Pauli
from qiskit.aqua.operators import WeightedPauliOperator
from qiskit.aqua.components.variational_forms import VariationalForm
from qiskit import execute


def generate_pauli_Z(idx, n, verbose=False):
    """
    n-qubit operator with Pauli Z located at idx
    This operator is needed for the smoothness Hamiltonian if idx has 2 elements
    This operator is needed for the value Hamiltonian if idx has 1 element
    """
    zeros = [0] * n
    zmask = [0] * n
    for i in idx: zmask[i] = 1
    a_z = np.asarray(zmask, dtype=np.bool)
    a_x = np.asarray(zeros, dtype=np.bool)
    if verbose:
        print(a_z, a_x)
    return Pauli(a_z, a_x)

def controlled_ry(circuit, theta, q_control, q_target):
    """
    Circuit needed for the entangling Ansatz
    Note that this function does not return anything,
    gates are added automatically when called
    """
    circuit.u3(theta / 2, 0, 0, q_target)
    circuit.cx(q_control, q_target)
    circuit.u3(-theta / 2, 0, 0, q_target)
    circuit.cx(q_control, q_target)

def print_grid(grid):
        """
        Pretty prints grid.
        """
        print("Grid:")
        for row in grid:
            for element in row:
                if element < 0:
                    print("{} ".format(element), end='')
                else:
                    print(" {} ".format(element), end='')
            print()

def build_zeta_operators(n):
    zetas = []
    for i in range(n):
        zetas.append(WeightedPauliOperator([(1.0, generate_pauli_Z([i], n))]))
    return zetas

def read_ground_state(result_matrix):
    """
    Takes in result['aux_ops'][0] from VQE algo output, i.e. measurement results
    in Pauli Z basis and converts to the corresponding ground state.
    In the future map the indexing to the actual pit coordinate using a dictionary.
    """
    ground_state = np.zeros(result_matrix[:,0].shape)
    for i in range(result_matrix[:,0].shape[0]): 
        ground_state[i] = np.around((1 - result_matrix[i,0]) / 2)
    #print(result_matrix[:,0])
    #exit()
    #ground_state = np.around((1 - result_matrix[:,0]) / 2)
    return ground_state

def read_masks(file_path):
    """
    Masks should be of size n x n and be written without empty lines in
    between into a txt file. Returns numpy.ndarray stacked with masks.
    """
    with open(file_path, 'r') as f:
        l = [[int(num) for num in line.split(' ')] for line in f]
    # Number of rows in the txt
    m = len(l)
    # Number of columns in the txt
    n = len(l[0])
    # Number of masks
    num_masks = int(m / n)
    mask_array = np.reshape(l, (num_masks, n, n))
    return mask_array

def make_fragments(mask_array):
    """
    Returns all fragment sites as a list of lists.
    """
    fragments = []
    for mask in mask_array:
        coordinates = np.argwhere(mask)
        idx = list([tuple(coordinate) for coordinate in coordinates])
        fragments.append(idx)
    return fragments

def count_fragment_params(fragment):
    """
    Given a fragment (a list of tuples), counts the number of parameters in
    the Ansatz, i.e. VariationalForm.
    """
    parent_count = 0
    for point in fragment:
        x, y = point
        if (x-1, y-1) in fragment: parent_count += 1
        if (x-1, y) in fragment: parent_count += 1
        if (x-1, y+1) in fragment: parent_count += 1
    point_count = len(fragment)
    total_count = point_count + parent_count
    return total_count

def combine_frag_results(results):
    """
    Combines all results from different fragments.
    """
    result_total = copy.deepcopy(results[0])
    for i in range(len(results)-1):
        result_total.grid += results[i+1].grid
        result_total.min_val += results[i+1].min_val
        result_total.elapsed_time += results[i+1].elapsed_time
        result_total.opt_params = np.append(result_total.opt_params, results[i+1].opt_params)
        result_total.ground_state = np.append(result_total.ground_state, results[i+1].ground_state)
    return result_total

def calculate_initial_expectation(variational_form, parameters, aux_op, statevector_mode=True):
    eval_circs = aux_op.construct_evaluation_circuit(wave_function=variational_form.construct_circuit(parameters), statevector_mode=statevector_mode)
    if statevector_mode:
        backend = BasicAer.get_backend('statevector_simulator')
    else:
        backend = BasicAer.get_backend('qasm_simulator')
    job = execute(eval_circs, backend, shots=1024)
    expectation_value, standard_deviation = aux_op.evaluate_with_result(job.result(), statevector_mode=statevector_mode)
    # Results are real anyway, we can discard the imaginary parts
    expectation_value = expectation_value.real
    standard_deviation = standard_deviation.real
    return expectation_value, standard_deviation



#Class for our variational form
class OpenPit_VarForm(VariationalForm):
    def __init__(self, num_qubits, num_layers, problem, num_params):
        super().__init__()
        self._configuration  = {'name': 'var_form_open_pit'}
        self._problem = problem
        self._num_qubits = num_qubits
        self._n_qubits = num_qubits
        print("N QUBITS ",self._n_qubits)
        self._num_orbitals = num_qubits
        self._num_layers = num_layers
        self._num_parameters = num_params
        self._bounds = [(-np.pi,np.pi)]*self._num_parameters

    def construct_circuit(self, parameters, q=None):
        """
		Builds a variational circuit using the form Mario has
		in small_qiskit_script. At each layer, each qubit gets
		a single RY rotation based on some paramter, and then
		gets entangled with the qubits who are its parents
		in a controlled Y operation.
		"""
        if q==None:
            q = QuantumRegister(self._n_qubits,name='q')
        circuit = QuantumCircuit(q)
        sites = self._problem._reachable_points
        param_ind = 0
        for i in range(self._num_layers):
            for point, point_ind in sites.items():
                circuit.ry(parameters[param_ind], point_ind)
                param_ind += 1
                parents = self._problem.get_parents(point)
                for parent in parents:
                    parent_qubit = sites[parent]
                    controlled_ry(circuit, parameters[param_ind], point_ind, parent_qubit)
                    param_ind += 1
        return circuit


class Fragmented_VarForm(OpenPit_VarForm):
    """
    Variational form for general fragments.
    """
    def __init__(self, num_qubits, num_layers, problem, num_params, fragment):
        super().__init__(num_qubits, num_layers, problem, num_params)
        self._fragment = fragment

    def construct_circuit(self, parameters, q=None, shake=False):
        """
		Controlled Y rotations are discarded if a parent-child bond is broken.
        If the fragment contains parents within itself, then we preserve the
        corresponding controlled Y rotations.
		"""
        shake = True
        np.random.seed(765)
        if q==None:
            q = QuantumRegister(self._n_qubits,name='q')
        circuit = QuantumCircuit(q)
        param_ind = 0
        for i in range(self._num_layers):
            for point_ind, point in enumerate(self._fragment):
                # DO NOT USE THE COMMENTED LINE.
                # It turns out that clipping is much better than modulo.
                # Modulo causes a problem with calculate_initial_expectation
                # parameters[point_ind] = parameters[point_ind] % np.pi
                delta = np.pi / 4
                parameters[param_ind] = np.clip(parameters[param_ind], 0, np.pi)
                if shake:
                    if parameters[param_ind] == np.pi:
                        parameters[param_ind] -= np.random.uniform(0, np.pi/2)
                    if parameters[param_ind] == 0:
                        parameters[param_ind] += np.random.uniform(0, np.pi/2)
                children = self._problem.get_children(point)
                for child in children:
                    if child in self._fragment:
                        parameters[param_ind] = np.clip(parameters[param_ind], 0, np.pi/2 + delta)
                        if shake:
                            if parameters[param_ind] == np.pi/2 + delta:
                                parameters[param_ind] -= np.random.uniform(0, np.pi/4)
                            if parameters[param_ind] == 0:
                                parameters[param_ind] += np.random.uniform(0, np.pi/4)
                circuit.ry(parameters[param_ind], point_ind)
                param_ind += 1
                parents = self._problem.get_parents(point)
                for parent in parents:
                    if parent in self._fragment:
                        parent_ind = self._fragment.index(parent)
                        parameters[param_ind] = np.clip(parameters[param_ind], 0, np.pi/2 - delta)
                        if shake:
                            if parameters[param_ind] == np.pi/2 - delta:
                                parameters[param_ind] -= np.random.uniform(0, np.pi/4)
                            if parameters[param_ind] == 0:
                                parameters[param_ind] += np.random.uniform(0, np.pi/4)
                        controlled_ry(circuit, parameters[param_ind], point_ind, parent_ind)
                        param_ind += 1
        return circuit
