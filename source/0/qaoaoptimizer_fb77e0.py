# https://github.com/incud/GraphEditDistance/blob/25fc7626cc29582f2a661e90e8695e3bf0c6c0c5/old/QaoaOptimizer.py
from PyQuboOptimizer import PyQuboOptimizer
from qiskit import Aer
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import QAOA
from qiskit.optimization import QuadraticProgram
from qiskit.algorithms.optimizers.cobyla import COBYLA
from qiskit.optimization.applications.ising.common import sample_most_likely


class QaoaOptimizer(PyQuboOptimizer):

    def __init__(self, hamiltonian, p=1, max_iter=1000):
        super(QaoaOptimizer, self).__init__(hamiltonian)
        self.map_names = {}
        self.variables = []
        self.p = p
        self.max_iter = max_iter
        self.sizes = None
        self.depths = None

    def replace_variable_name(self, name):
        new_name = name.replace("][", "_").replace("[", "_").replace("]", "")
        self.map_names[name] = new_name
        return new_name

    def get_operator(self):
        model = self.H.compile()
        qubo, offset = model.to_qubo()
        # rename variables from x[0] to x_0
        qubo = {(self.replace_variable_name(x1), self.replace_variable_name(x2)): v
                for (x1, x2), v in qubo.items()}
        self.variables = sorted(self.map_names.values())
        # create DOcplex quadratic program
        cplex_program = QuadraticProgram()
        # add the variables
        for x in self.variables:
            cplex_program.binary_var(x)
        # add the QUBO problem to DOcplex
        cplex_program.minimize(quadratic=qubo, constant=offset)
        operator, offset = cplex_program.to_ising()
        return operator

    def vector_to_sample(self, vector):
        inv_map = {v: k for k, v in self.map_names.items()}
        sample = {}
        for i in range(len(vector)):
            var = inv_map[self.variables[i]]
            sample[var] = int(vector[i])
        return sample

    def solve(self):
        operator = self.get_operator()
        optimizer = COBYLA(maxiter=self.max_iter)
        simulator = QuantumInstance(Aer.get_backend('qasm_simulator'), shots=2048)
        qaoa = QAOA(operator, quantum_instance=simulator, optimizer=optimizer, p=self.p)
        result = qaoa.compute_minimum_eigenvalue()
        vector = sample_most_likely(result.eigenstate)
        best_sample = self.vector_to_sample(vector)

        self.best_sample = best_sample.sample
        self.best_energy = best_sample.energy
        return self.best_sample, self.best_energy

    def get_record_result(self):
        return {
            "solver": "qaoa",
            "best_sample": self.best_sample,
            "best_energy": self.best_energy,
            "p": self.p,
            "max_iter": self.max_iter
        }
