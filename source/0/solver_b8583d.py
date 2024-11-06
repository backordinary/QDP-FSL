# https://github.com/quantum-mining/quantum-mining-publication/blob/606bda884426474a12bccac69d0aad55804207ef/mine_solver/solver.py
import numpy as np
import sys
from utils import *
from problem import OpenPitMiningProblem
from config import Configuration
from result import Result

from qiskit import QuantumRegister, QuantumCircuit, Aer, BasicAer
from qiskit.aqua import QuantumInstance, aqua_globals
from qiskit.aqua.algorithms                        import VQE
#from qiskit.aqua.algorithms.adaptive import VQE
#from qiskit.aqua.algorithms.classical import ExactEigensolver
from qiskit.aqua.algorithms                        import ExactEigensolver
#from qiskit.aqua.components.optimizers             import L_BFGS_B,COBYLA,CG
from qiskit.aqua.components.optimizers import L_BFGS_B, CG, SPSA, SLSQP
from qiskit import IBMQ
from qiskit.ignis.mitigation.measurement import (complete_meas_cal,tensored_meas_cal,CompleteMeasFitter,TensoredMeasFitter)

import networkx as nx
#import pseudoflow

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


class Solver():
    """
    Abstract solver for an open pit mining circuit.
    """

    def __init__(self, problem, options):
        self._problem = problem
        self.options = options
        self._n_qubits = options['num_qubits']

    def solve(self):
        """
        Uses internal data (circuit, potentially params)
        and returns a solution to the open pit mining problem
        in the form of a mask
        """
        pass


class ExactSolver(Solver):
    """
    If the problem size is small enough, this solver computes the theoretical
    best solution.
    """

    def __init__(self, problem, options):
        super().__init__(problem, options)

    def solve(self, verbose=False):
        """
        Solves the problem using ExactEigensolver.
        """
        # Construct hamiltonian
        if verbose:
            print("------------------------------------")
            print("Building Hamiltonian...", end="")
        n, qubit_op, offset = self._problem.build_hamiltonian(verbose=verbose)
        if verbose:
            print(" Done!")

        # Create auxiliary operators to perform final measurements on each cell
        zeta_ops = build_zeta_operators(n)

        # Create and run ExactEigensolver
        if verbose:
            print("Running ExactEigensolver...", end="")
        result = Result(self._problem.get_dims(), verbose=verbose)
        algo = ExactEigensolver(qubit_op, aux_operators=zeta_ops)
        solver_out = algo.run()

        output = {}
        output['aux_ops'] = solver_out['aux_ops']
        output['min_val'] = solver_out['energy']
        output['opt_params'] = None

        result.add_solver_output(output)
        if verbose:
            print(" Done!")
        # Print results
        if verbose:
            print("------------------------------------")
            result.print_results()
        return result


class PseudoflowSolver(Solver):
    """
    Solver which implements the pseudoflow algorithm.
    Inspired by: https://www.youtube.com/watch?v=nt8XU2yONJc
    """

    def __init__(self, problem, options, MAX_FLOW=1000000):
        super().__init__(problem, options)
        self.MAX_FLOW = MAX_FLOW

    def build_graph(self):
        """
        Turns the problem graph into a graph that can be used
        in the pseudoflow algorithm. Uses networkx DiGraph
        for the directed graph.
        """
        grid = self._problem._grid
        all_nodes = self._problem.get_all_sites()

        G = nx.DiGraph()
        source = -1
        sink = len(all_nodes)

        for coord, ind in all_nodes.items():
            parents = self._problem.get_parents(coord)
            for parent in parents:
                # for each parent, add an infinite capacity edge
                parent_ind = all_nodes[parent]
                G.add_edge(ind, parent_ind, const=self.MAX_FLOW)

            # if positive, connect to source, else connect to sink
            if grid[coord] >= 0:
                G.add_edge(source, ind, const=grid[coord])
            else:
                G.add_edge(ind, sink, const=-grid[coord])
        return G, source, sink

    def solve(self, verbose=False):
        if verbose:
            print("Building graph from grid.")
        graph, source, sink = self.build_graph()
        if verbose:
            print("Solving using pseudoflow.")

        breakpoints, cuts, info = pseudoflow.hpf(
            graph,
            source,
            sink,
            const_cap='const')

        # fashion result into way similar to VQE so Result
        # class can handle it.
        result = Result(self._problem.get_dims(), verbose=verbose)
        output = {}
        ground_state = [value[0] for _, value in cuts.items()][1:-1]
        # this is for the weird result processing which is expecting something
        # in {-1, +1}
        output['aux_ops'] = [-(np.array(ground_state)[:, np.newaxis]*2-1)]
        grid = self._problem._grid
        # Fix the min val, currently it's the total value of digging
        output['min_val'] = sum(grid[grid.nonzero()]*ground_state)
        output['opt_params'] = None
        result.add_solver_output(output)
        if verbose:
            print("------------------------------------")
            result.print_results()
        return result


class VariationalSolver(Solver):
    def __init__(self, problem, options, parameters=None):
        super().__init__(problem, options)
        self._parameters = parameters
        self._num_layers = options['num_layers']

    def build_initial_params(self, num_qubits, initial_value=np.pi/2):
        """
        If initial params are not provided, this function will
        populate self.parameters with the correct parameters.
        """
        self._parameters = []
        sites = self._problem._reachable_points
        for i in range(self._num_layers):
            for point, point_ind in sites.items():
                self._parameters.append(initial_value)
                parents = self._problem.get_parents(point)
                for parent in parents:
                    self._parameters.append(initial_value)

    def choose_optimizer(self):
        if(self.options['optimizer'] == 'L_BFGS_B'):
            self.optimizer = L_BFGS_B(
                maxiter=self.options['max_iter'])#, iprint=1001)
        elif(self.options['optimizer'] == 'CG'):
            self.optimizer = CG(maxiter=self.options['max_iter'])
        elif(self.options['optimizer'] == 'SLSQP'):
            self.optimizer = SLSQP(maxiter=self.options['max_iter'])
        elif(self.options['optimizer'] == 'SPSA'):
            self.optimizer = SPSA()
        else:
            self.optimizer = SPSA()

    def solve(self, verbose=False, backend_option="simulator", filename=None, params_filename=None):
        """
        Variationally solves the problem using VQE. The circuit
        should already be built.
        """
        # Construct hamiltonian
        self.choose_optimizer()
        if verbose:
            print("------------------------------------")
            print("Building Hamiltonian...", end="")
        n, qubit_op, offset = self._problem.build_hamiltonian(verbose=verbose)
        if verbose:
            print(" Done!")

        # Create auxiliary operators to perform final measurements on each cell
        zeta_ops = build_zeta_operators(n)

        # Create initial parameters, variational form based on input options
        if verbose:
            print("Building Circuit...", end="")
        if self._parameters is None:
            self.build_initial_params(n)
        self.var_form = OpenPit_VarForm(
            self._n_qubits, self._num_layers, self._problem, len(self._parameters))
        if verbose:
            print(" Done!")

        # Create and run VQE
        if verbose:
            print("Running VQE...", end="")
        self.result = Result(self._problem.get_dims(), verbose=verbose, filename=filename,params_filename=params_filename)
        self.algo = VQE(qubit_op, self.var_form, self.optimizer,
                        initial_point=self._parameters,
                        aux_operators=zeta_ops,
                        callback=self.result.iteration_callback)

        # Set appropriate backend and run the algorithm
        if backend_option == "hardware":
            IBMQ.load_account()
            #provider = IBMQ.get_provider(hub='ibm-q-academic', group='stanford', project='gretzky-sandbox')
            provider = IBMQ.get_provider(hub='ibm-q-internal', group='deployed', project='default')
            self.backend = provider.get_backend('ibmq_paris')
            solver_out = self.algo.run(QuantumInstance(self.backend, seed_simulator=aqua_globals.random_seed,
                                                       seed_transpiler=aqua_globals.random_seed, shots=8192, skip_qobj_validation=False,
                                                       initial_layout = [22,24,26,25], optimization_level = 0))
        elif backend_option == "hardware_with_em":
            IBMQ.load_account()
            #provider = IBMQ.get_provider(hub='ibm-q-academic', group='stanford', project='gretzky-sandbox')
            provider = IBMQ.get_provider(hub='ibm-q-internal', group='deployed', project='default')
            self.backend = provider.get_backend('ibmq_paris')
            # trying VQE with converged initial parameters
            solver_out = self.algo.run(QuantumInstance(self.backend, seed_simulator=aqua_globals.random_seed,
                                                       seed_transpiler=aqua_globals.random_seed, shots=8192, measurement_error_mitigation_cls=CompleteMeasFitter, skip_qobj_validation=False,initial_layout = [22,24,26,25], optimization_level = 0))
        elif backend_option == "simulator":
            self.backend = Aer.get_backend(self.options['backend'])
            solver_out = self.algo.run(QuantumInstance(self.backend, seed_simulator=aqua_globals.random_seed,
                                                       seed_transpiler=aqua_globals.random_seed, shots=8192))

        # Set appropriate backend and run the algorithm
        #self.backend = BasicAer.get_backend(self.options['backend'])
        #solver_out = self.algo.run(QuantumInstance(self.backend, seed_simulator=aqua_globals.random_seed,
        #                                              seed_transpiler=aqua_globals.random_seed, shots=1000))

        self.result.add_solver_output(solver_out)

        if verbose:
             print(" Done!")
         # Print results
        if verbose:
            print("------------------------------------")
            self.result.print_results()
        return self.result


class FragmentationSolver(Solver):
    """
    Generalized fragmented VQE solver.
    """

    def __init__(self, problem, options, parameters=None):
        super().__init__(problem, options)
        self._parameters = parameters  # Should be a list of lists
        self._num_layers = options['num_layers']
        self._masks_file_path = options['mask']
        self._num_loops = options['frag_num_loops']
        self.all_data = None

    def build_initial_params(self, fragment, initial_value=np.pi/2):
        """
        If initial params are not provided, this function will
        populate self.parameters with the correct parameters.
        """
        # Setting all initial values to the boundary values is problematic
        #delta = 0.3
        #initial_value = np.clip(initial_value, 0 + delta, np.pi - delta)
        f_num_params = count_fragment_params(fragment)
        return [initial_value] * f_num_params
        # return np.random.rand(f_num_params) * np.pi

    def choose_optimizer(self):
        if(self.options['optimizer'] == 'L_BFGS_B'):
            self.optimizer = L_BFGS_B(
                maxiter=self.options['max_iter'], iprint=1001)
        elif(self.options['optimizer'] == 'CG'):
            self.optimizer = CG(maxiter=self.options['max_iter'])
        elif(self.options['optimizer'] == 'SLSQP'):
            self.optimizer = SLSQP(maxiter=self.options['max_iter'])
        elif(self.options['optimizer'] == 'SPSA'):
            self.optimizer = SPSA(max_trials=self.options['max_iter'])
        else:
            self.optimizer = SPSA(max_trials=self.options['max_iter'])

    def solve(self, verbose=False):
        """
        Divides the variational problem into fragments.
        """
        # Optimizer
        self.choose_optimizer()

        # Set backend
        self.backend = BasicAer.get_backend(self.options['backend'])

        # Generate fragments, as a list of lists
        # Each list contains the site indices as tuples
        # E.g. fragments = [[(0,0), (0,1), (0,2)], [(1,1)]]
        mask_array = read_masks(self._masks_file_path)
        fragments = make_fragments(mask_array)
        num_frags = len(fragments)

        # Result for fragment i in jth run will be stored in all_results[j][i]
        # Same to keep track of parameters and expectations across all runs
        all_results = []
        all_parameters = []
        all_expectations = []

        if verbose:
            print("------------------------------------")
            print("Building Circuit...")

        # Some variables and operators needed for later
        # Each of these is a list of lists where list[i]
        # contains the corresponding information on i-th fragment
        n_qubits = []  # Fragment sizes, e.g. [3, 1]
        frags_num_params = []  # Number of parameters in each fragment, e.g. [3, 1]
        frags_zeta_ops = []  # For final measurements after VQE, e.g. [[IIZ, IZI, ZII], [Z]]
        var_forms = []  # E.g. [var_form_f1, var_form_f2]
        parameters = []  # E.g. [[1.57, 1.57, 1.57], [1.57]]
        expectations = []  # E.g. [[0, 0, 0], [0]]
        results = []

        for fragment in fragments:
            n = len(fragment)
            f_num_params = count_fragment_params(fragment)
            n_qubits.append(n)
            frags_num_params.append(f_num_params)
            frags_zeta_ops.append(build_zeta_operators(n))
            var_forms.append(Fragmented_VarForm(
                n, self._num_layers, self._problem, f_num_params, fragment))
            if self._parameters is None:
                parameters.append(self.build_initial_params(fragment))
            # Fill "expectations" with dummy numbers and update later
            expectations.append([0] * n)
            results.append([])

        if self._parameters is None:
            self._parameters = parameters

        # Compute the initial expectation of every cell before the first VQE run
        # E.g. [[1.1102230246251565e-16, 1.3877787807814457e-16, 1.7191788613864501e-16], [2.220446049250313e-16]]
        for i, fragment in enumerate(fragments):
            n = len(fragment)
            for j in range(n):
                exp, std_dev = calculate_initial_expectation(
                    var_forms[i], self._parameters[i], frags_zeta_ops[i][j])
                expectations[i][j] = exp

        if verbose:
            print("Initial parameters:", self._parameters)
            print("Initial expectations:", expectations)
            print("Total number of Ansatz parameters is",
                  np.sum(frags_num_params))
            print("Done!")

        for i in range(self._num_loops):
            for j, fragment in enumerate(fragments):

                # Construct Hamiltonian
                if verbose:
                    print("------------------------------------")
                    print("IN LOOP", i+1, "OF", self._num_loops)
                    print("Building Hamiltonian for Fragment {} of {}...".format(
                        j+1, num_frags))
                # using the general fragment Hamiltonian builder function :)))
                qubit_op, offset = self._problem.build_fragment_hamiltonian(
                    fragments, j, expectations)

                if verbose:
                    print("Done!")

                # Create and run VQE
                if verbose:
                    print("Running VQE for Fragment {} of {}...".format(
                        j+1, num_frags))
                self.result = Result(self._problem.get_dims(), verbose=verbose, filename=filename,params_filename=params_filename)
                self.algo = VQE(qubit_op, var_forms[j], self.optimizer,
                                initial_point=self._parameters[j],
                                aux_operators=frags_zeta_ops[j],
                                callback=self.result.iteration_callback)

                # Run the algorithm
                solver_out = self.algo.run(QuantumInstance(self.backend, seed_simulator=aqua_globals.random_seed,
                                                           seed_transpiler=aqua_globals.random_seed, shots=1000))

                # Expected value (vector) to be provided to other Hamiltonian in the next run
                expectations[j] = solver_out['aux_ops'][0][:, 0]

                # Update result
                self.result.add_solver_output(solver_out, fragment)
                self._parameters[j] = self.result.opt_params
                results[j] = self.result

            # There's a bug with the append function, it also replaces the old information inside!!!
            all_results.append(results)
            all_expectations.append(expectations)
            all_parameters.append(self._parameters)

        # Gather all results together
        self.all_data = {'all_results': all_results,
                         'all_expectations': all_expectations, 'all_parameters': all_parameters}
        self.result_total = combine_frag_results(results)

        if verbose:
            # print(self.all_data)
            print("Done!")
            for j, fragment in enumerate(fragments):
                print("------------------------------------")
                results[j].print_results()
            print("------------------------------------")
            self.result_total.print_results()

        return self.all_data, self.result_total


if __name__ == "__main__":
    config_filename = '../configs/config1.json'
    configuration = Configuration(config_filename)
    problem = OpenPitMiningProblem(configuration.get_problem_options())
    solver_options = configuration.get_solver_options()
    #e_solver = ExactSolver(problem, solver_options)
    #e_result = e_solver.solve(verbose=True)
    #p_solver = PseudoflowSolver(problem, solver_options)
    #p_result = p_solver.solve(verbose=True)
    v_solver = VariationalSolver(problem, solver_options)
    #, parameters = [2.42678417,2.00978447,0.51443224,3.12972106,1.12478717,1.13880863,2.44631943])
    v_result = v_solver.solve(verbose=True, backend_option="hardware_with_em")
    #f_solver = FragmentationSolver(problem, solver_options, parameters=None)
    #f_data, f_result_total = f_solver.solve(verbose=True)
