# https://github.com/bowrango/quantum-circuits/blob/7b6941b29df7f79179e1962ba637b4f0be031718/aqc.py
import numpy as np

from qiskit.circuit import ParameterVector
from qiskit.algorithms.optimizers import L_BFGS_B
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from templates import cartan_network, spin_network, sequential_network, get_connectivity
from objective import Objective

# Approximate Circuit Compiling
class AQC(Objective):
    def __init__(self, target: np.ndarray, template: str, depth: int, connectity: str) -> None:
        super().__init__

        self.target = target
        self.template = template
        self.connectivity = connectity
        self._num_qubits = int(np.log2(len(target)))

        # template and depth determine number of parameters
        if template == "cart":
            # fixed depth and connectivity
            cnots = cartan_network(self._num_qubits) 
        if template == "spin":
            # fixed connectivity
            cnots = spin_network(self._num_qubits, depth) 
        if template == "sequ":
            links = get_connectivity(self._num_qubits, connectivity=connectity)
            cnots = sequential_network(self._num_qubits, links, depth)
        
        self.num_cnots = cnots.shape[1]
        self._cnots = cnots

    # construct the circuit ansatz
    def build(self):
    
        # 3 initial rotations on each qubit plus 4 for each CNOT unit
        params = ParameterVector('theta', length=4*self.num_cnots+3*self._num_qubits)
        qc = QuantumCircuit(self._num_qubits)

        for q in range(self._num_qubits):
            p = 3*q
            qc.rz(params[0+p], q)
            qc.ry(params[1+p], q)
            qc.rz(params[2+p], q)

        start = 3*self._num_qubits
        for l,(c,t) in enumerate(zip(self._cnots[0], self._cnots[1])):
            p = range(start + (4*l), start + 4*(l+1)) 
            qc.cx(c,t)
            qc.ry(params[p[0]], c)
            qc.rx(params[p[1]], c)
            qc.ry(params[p[2]], t)
            qc.rz(params[p[3]], t)
        
        self.ansatz = qc 
        self.params = params
        print(self.ansatz)

    # main optimization routine    
    def compile(self) -> None:

        self.build()

        angles = np.random.uniform(0, 2*np.pi, size=len(self.params))

        optimizer = L_BFGS_B(maxiter=1000, iprint=10, ftol=1e-9)
        result = optimizer.minimize(
            fun=self.objective,
            x0=angles,
            jac=self.gradient
        )

        self.ansatz.assign_parameters({self.params: result.x}, inplace=True)
        self.compiled_matrix = Operator(self.ansatz).data

    def compress(self) -> None:
        return




