# https://github.com/MarcoArmenta/QAOA/blob/9c25a88e3e1c72886fde869b1657b0f734bdf0e9/QAOA/QAOA.py
import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
import operator

from qiskit.algorithms.optimizers import COBYLA, SLSQP, ADAM, GradientDescent, SPSA, QNSPSA
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.circuit.library import TwoLocal
from qiskit.circuit import ParameterVector
from qiskit import QuantumCircuit, Aer
from qiskit.aqua import QuantumInstance
from qiskit.algorithms import QAOA

from qiskit.algorithms.minimum_eigen_solvers.vqe import VQEResult

_offset_ = 0

#'qnspsa': QNSPSA() - requires fidelity



class QAOA_01:
    __optimizers = {'adam': ADAM(), 'cobyla': COBYLA(), 'slsqp': SLSQP(), 'gradientdescent': GradientDescent(),
                      'spsa': SPSA()}

    history = {'beta': [], 'gamma': [], 'energy': []}

    def __init__(self, graph : nx.Graph, num_layers=1, optimizer='spsa', epochs=10,
                 name='maxcut', ansatz_type=None, qiskit_runtime=False, backend='aer_simulator',
                 shots=10, reps=1):

        try:
            self.backend = Aer.get_backend(backend, shots=shots)
            self.q_instance = QuantumInstance(backend=self.backend, shots=shots)

        except:
            print('Using aer_simulator as backend')
            self.backend = Aer.get_backend('aer_simulator', shots=shots)
            self.q_instance = QuantumInstance(backend=self.backend, shots=shots)

        if qiskit_runtime:
            # TODO: add runtime script
            TypeError('You need to code something to use qiskit runtime')

        else:
            try:
                self.optimizer = QAOA_01.__optimizers[optimizer.lower()]
            except:
                print('Using ADAM optimizer.')
                self.optimizer = QAOA_01.__optimizers['adam']

            self.epochs = epochs
            self.graph = graph
            self.num_qubits = graph.number_of_nodes()
            self.num_layers = num_layers
            self.name = name
            self.ansatz_type = ansatz_type
            self.reps = reps

            self.quadratic_program = None
            self.ansatz = None
            self.circuit = None
            self.gammas = None
            self.betas = None
            self.result = None

            self.__build_ansatz()
            print('Making quadratic program')
            self.__make_quadratic_program()
            print('Building circuit')
            self.__build_circuit()
            print('Solving maxcut instance')
            self.__solve_maxcut()
            print('Result obtained by the optimized model')
            dictionary = self.result.min_eigen_solver_result.eigenstate
            # Key with maximum value in dictionary (the key is a string)
            result = max(dictionary.items(), key=operator.itemgetter(1))[0]
            print(result)

            self.print_graph(result)



    def __build_ansatz(self):
        pass
        # TODO: add more ansatze
        #if self.ansatz_type is None:
        #    ansatz = TwoLocal(self.num_qubits, ['rz'], 'cx', 'linear', reps=3)
        #else:
        #    ansatz = TwoLocal(self.num_qubits, ['rz'], 'cx', 'linear', reps=1)

        #self.ansatz = ansatz

    def __make_quadratic_program(self):
        qp = QuadraticProgram(self.name)
        weights = nx.adj_matrix(self.graph)

        quadratic = np.zeros((self.num_qubits, self.num_qubits))
        linear = np.zeros(self.num_qubits)

        for i in range(self.num_qubits):
            for j in range(self.num_qubits):
                quadratic[i, j] -= weights[i, j]
                linear[i] += weights[i, j]

        for i in range(self.num_qubits):
            var = 'x_' + str(i)
            qp.binary_var(name=var)

        qp.maximize(quadratic=quadratic, linear=linear)

        self.quadratic_program = qp

        global _offset_
        _offset_ = 0.25 * self.quadratic_program.objective.quadratic.to_array(symmetric=True).sum() \
                      + 0.5 * self.quadratic_program.objective.linear.to_array().sum()


    def __build_circuit(self):
        quadratic = self.quadratic_program.objective.quadratic.to_array(symmetric=True)

        circuit = QuantumCircuit(self.num_qubits, self.num_qubits)

        circuit.h(range(self.num_qubits))

        gamma = ParameterVector('gamma', self.num_layers)
        beta = ParameterVector('beta', self.num_layers)

        for i in range(self.num_layers):
            for j1 in range(self.num_qubits):
                for j2 in range(self.num_qubits):
                    if quadratic[j1, j2] != 0:
                        circuit.cx(j1, j2)
                        circuit.rz(gamma[i] * quadratic[j1, j2]/2.0, j2)
                        circuit.cx(j1, j2)
            for j in range(self.num_qubits):
                circuit.rx(2 * beta[i], j)

        self.circuit = circuit

    def __solve_maxcut(self):
        init_point = np.random.random(2*self.reps)
        print('Build QAOA qiskit')
        qaoa = QAOA(optimizer=self.optimizer, quantum_instance=self.backend, reps=self.reps,
                    initial_point=init_point, callback=QAOA_01.call_back)
        print('Build Eigen Optimizer')
        eigen_optimizer = MinimumEigenOptimizer(min_eigen_solver=qaoa)
        print('Solving quadratic program')
        self.result = eigen_optimizer.solve(self.quadratic_program)

    @staticmethod
    def call_back(eval_count, params, mean, std_dev):
        QAOA_01.history['beta'].append(params[1])
        QAOA_01.history['gamma'].append(params[0])
        QAOA_01.history['energy'].append(-mean + _offset_)




    def print_graph(self, x):
        default_axes = plt.axes(frameon=False)
        pos = nx.spring_layout(self.graph)

        colors = ['r' if x[i] == '1' else 'b' for i in range(len(x))]

        nx.draw_networkx(self.graph, node_color=colors, node_size=600, alpha=1, ax=default_axes, pos=pos)
        plt.show()

    def maxcut(self, lr=0.01, samples=10, epochs=20, verbose=False):
        pass