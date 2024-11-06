# https://github.com/AnnaMHua/QAOA_for_partition/blob/f2d226f2faa082709a201d86fcf336e9c7bb8db0/QAOA.py
import numpy as np

from qiskit import Aer
from qiskit import QuantumRegister, QuantumCircuit, execute

from scipy.optimize import  minimize
from numpy import pi

from qiskit.optimization.applications.ising.common import read_numbers_from_file


class partitionQAOA:

    def __init__(self, number_list, layer, backend = "statevector_simulator", NUM_SHOTS = 1000, method = 'COBYLA'):
        '''
        QAOA for number partitioning.
        :param number_list: a set of numbers to be partitioned
        :param layer: number of QAOA layers
        :param backend: the backend used for quantum circuit
        :param NUM_SHOTS: number of shots in quantum measurement
        :param method: the optimization method
        '''

        self.p = layer
        self.set = number_list
        self.nqubits = len(number_list)
        self.backend = backend
        self.NUM_SHOTS = NUM_SHOTS
        self.method = method

    def QAOAcircuit(self,parameters):
        '''
        build QAOA circuit with input parameters
        :param parameters: a size-(layer,2) array for gamma and beta in QAOA
        :return: a qiskit QuantumCircuit instance for QAOA
        '''

        qc = QuantumCircuit(QuantumRegister(self.nqubits))

        # start from a superposition state
        qc.h(range(self.nqubits))

        parameters = parameters.reshape((self.p, 2))
        for layer in range(self.p):
            # one layer of QAOA circuit
            gamma = parameters[layer, 0]
            beta = parameters[layer, 1]

            # add U_C: unitary evolution under Hamiltonian
            for i in range(self.nqubits - 1):
                for j in range(i + 1, self.nqubits):
                    qc.cx(i, j)
                    qc.rz(-4 * self.set[i] * self.set[j] * gamma, j)
                    qc.cx(i, j)
                    qc.barrier()

            # add U_B:the mixer
            qc.rx(beta, range(self.nqubits))

        return qc

    def expectation(self,counts):
        '''
        build function to calculate the expectation of H for number partitioning problem
        :param counts: a quantum state distribution
        :return: expectation value of a distribution
        '''

        expect = 0
        for config, prob in counts.items():
            # for each eigenstate we can calculate its energy

            # reverse bits due to the qiskit convention
            x = list(map(int, config))[::-1]
            energy = 0
            for i in range(self.nqubits):
                for j in range(i):
                    energy += 2 * self.set[i] * self.set[j] * x[i] * x[j]
            expect += energy * prob
        return expect

    def cost_func(self,parameters):
        '''
        build the final cost function, evaluate the function using quantum backend
        :return: the expectation vale of the final measurement
        '''
        qc = self.QAOAcircuit(parameters)

        final_distribution = execute(qc, Aer.get_backend(self.backend),
                                     shots=self.NUM_SHOTS).result().get_counts()

        return self.expectation(final_distribution)

    def optimization_output(self):
        '''
        perform classical optimization in QAOA circuit
        :return: OptimizeResult object in Scipy
        '''

        # initialize parameters at random value in(0,2pi)
        params = 2 * pi * np.random.rand(self.p * 2)

        # optimization
        ret = minimize(fun=self.cost_func, x0=params, method=self.method, tol=0.0001, options={'maxiter': 5000})
        # print("Layer :: ",layer, " Optimized cost::", cost, "  Success:",ret.success)
        # print("Optimized parameter::", ret.x)
        return ret
