# https://github.com/paulmatt67/quantum_packer_public/blob/b37d57d4a1682579dd4025115215d0ef0f2eb8bc/QUBO_SOLVER.py
from qiskit import BasicAer
from qiskit.aqua import aqua_globals, QuantumInstance
from qiskit.aqua.algorithms import QAOA, NumPyMinimumEigensolver
from qiskit.optimization.algorithms import MinimumEigenOptimizer, RecursiveMinimumEigenOptimizer
from qiskit.optimization import QuadraticProgram
from qiskit_methods.backends import *


class QUBO_SOLVER():

    def __init__(self, binary_variables, linear_coefficients, quadratic_coefficients, machine):
        """

        :param binary_variables: list of strings with the names of the binary variables
        :param linear_coefficients: list of value of the coefficients for each binary variable
        :param quadratic_coefficients: dictionary with key: value pairs of the form (var1, var2): coefficient
        :param machine: 'local_qasm_simulator', 'IBM-US', 'IBM-DE'
        """
        self.num_qubits_needed = len(binary_variables)
        self.machine = machine
        # if the machine chosen is qasm_simulator, IBM-US or IBM-DE
        if machine in ['local_qasm_simulator', 'IBM-US', 'IBM-DE']:
            # create a QUBO in qiskit
            qubo = QuadraticProgram()
            for b in binary_variables:
                qubo.binary_var(b)
            qubo.minimize(linear=linear_coefficients, quadratic=quadratic_coefficients)
            print(qubo.export_as_lp_string())
            self.qubo = qubo
        else:
            print('WARNING: solving a QUBO on the machine ' + machine + ' is not implemented !')

    def resolve(self, verbose=False):
        # Convert QUBO to Ising model
        op, offset = self.qubo.to_ising()
        if verbose:
            print('offset: {}'.format(offset))
            print('operator:')
            print(op)
        # Set random seed
        #aqua_globals.random_seed = 42
        # Set the quantum backend depending on the machine and number of qubits needed
        if self.machine == 'local_qasm_simulator':
            quantum_backend = BasicAer.get_backend('qasm_simulator')
        elif self.machine == 'simulator_mps':
            quantum_backend = BasicAer.get_backend('simulator_mps')
        else:
            if self.machine == 'IBM-DE':
                provider = load_providers_de()
            else:
                # use US providers by default
                provider = load_providers_us(project='member')
            num_qubits_per_backend_dict = show_num_qubits_for_provider(provider)
            try:
                quantum_backend_identifier = find_least_busy_backend(provider, n_qubits=self.num_qubits_needed)
            except:
                quantum_backend_identifier = find_simulator_backend(provider, n_qubits=self.num_qubits_needed)
            print('Solving QUBO with ' + str(self.num_qubits_needed) + ' qubits on ' + self.machine + '/' + quantum_backend_identifier + ' equipped with ' + str(num_qubits_per_backend_dict[quantum_backend_identifier]) + ' qubits')
            quantum_backend = provider.get_backend(quantum_backend_identifier)
        # Create quantum instance
        quantum_instance = QuantumInstance(quantum_backend,
                                           seed_simulator=aqua_globals.random_seed,
                                           seed_transpiler=aqua_globals.random_seed)
        # Solve quantum instance using QAOA
        qaoa_mes = QAOA(quantum_instance=quantum_instance)
        qaoa = MinimumEigenOptimizer(qaoa_mes)  # using QAOA
        qaoa_result = qaoa.solve(self.qubo)
        if verbose:
            print(qaoa_result)
        return qaoa_result.variables_dict



