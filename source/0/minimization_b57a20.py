# https://github.com/sidambhire/QCID-Project/blob/8d467d93ab55f51241c3d53bda3e71efaa74378a/lib/minimization.py
import math
import random
from numpy.random import choice

import numpy as np
from qiskit import Aer, transpile
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.aqua.algorithms.amplitude_amplifiers.grover import Grover
from qiskit.circuit.library import QuadraticForm

from qiskit.optimization.algorithms.optimization_algorithm import OptimizationAlgorithm
from qiskit.optimization.converters.quadratic_program_to_qubo import QuadraticProgramToQubo
from qiskit.optimization.problems.quadratic_program import QuadraticProgram

class minimization(OptimizationAlgorithm):
    
    def __init__(self, num_qb: int) -> None:
        
        self.var_qb = num_qb
        self.key_qb= None

    def get_compatibility_msg(self, problem: QuadraticProgram) -> str:
        """
        must define abstract method from super
        """
        return QuadraticProgramToQubo.get_compatibility_msg(problem)

    def get_operator(self, qr, problem) -> QuantumCircuit:
        """
        Create operator used in Grover search
        """
        quadratic = QuadraticForm(self.var_qb, problem.objective.quadratic.to_array(), problem.objective.linear.to_array(), problem.objective.constant, little_endian=False)
        operator = QuantumCircuit(qr)
        operator.h(list(range(self.key_qb)))
        operator.compose(quadratic, inplace= True)
        return operator

    def oracle(self, qr):
        """
        Define oracle for the minimization algorithm
        """
        oracle_reg = QuantumRegister(1, "oracle")
        oracle = QuantumCircuit(qr, oracle_reg)
        oracle.z(self.key_qb) #flip negatives

        def state(self, measure):
            """
            Define how to check for a good state for Grover search
            """
            key = measure[self.key_qb : self.key_qb + self.var_qb]
            return key[0] == '1'

        return oracle, state

    def solve(self, problem: QuadraticProgram):
        """
        Solving the minimization problem
        """
        qubo = self._convert(problem, QuadraticProgramToQubo()) #covert to Qubo 
        self.key_qb = len(qubo.objective.linear.to_array())

        solved = False
        opt_key = math.inf
        opt_value = math.inf
        threshold = 0
        constant = qubo.objective.constant
        num_var = len(qubo.variables)
        rotations = 0

        qr = QuantumRegister(self.key_qb + self.var_qb)
        oracle, state = self.oracle(qr)
        
        while not solved:
            m = 1
            improved = False
            qubo.objective.constant = constant - threshold
            operator = self.get_operator(qr, qubo)
            iter_not_improved = 0
            while not improved:
                iter_not_improved += 1
                rotation = random.randint(0, m - 1)
                rotations += rotation
                grover_circ = Grover(oracle, state_preparation = operator, 
                                    good_state= state).construct_circuit(rotation, True)

                simulator = Aer.get_backend('aer_simulator')
                circ = transpile(grover_circ, simulator)
                result = simulator.run(circ).result()
                temp = result.get_counts(circ)
                draw = choice(list(temp.keys()))
                outcome = draw[::-1]

                key = int(outcome[0:num_var], 2)
                value = outcome[num_var:num_var + self.var_qb]
                int_v = self._bin_to_int(value, self.var_qb) + threshold
                #print(key, " ", value, " ", int_v)

                if int_v < opt_value: #improvement found
                    opt_key = key
                    opt_value = int_v
                    improved = True
                    threshold = opt_value
                else:
                    m = int(np.ceil(m * 8/7))
                    if iter_not_improved > 10:
                        improved = True
                        solved = True
        answer = "{0:b}".format(opt_key)
        for i in range(len(answer)):
            print(i+1, "th variable value is ", answer[i])
        return 

    @staticmethod
    def _bin_to_int(v: str, num_value_bits: int) -> int:
        """Converts a binary string of n bits using two's complement to an integer."""
        if v.startswith("1"):
            int_v = int(v, 2) - 2 ** num_value_bits
        else:
            int_v = int(v, 2)

        return int_v