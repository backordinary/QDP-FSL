# https://github.com/Red2Je/IBMInternship/blob/3bd7034c1a4245c134b44c682c549491dfed3ce6/WSL%20works/testgpu.py
from qiskit import Aer
from qiskit.providers.aer import AerSimulator
from qiskit.opflow import X, Z, I
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import SLSQP
from qiskit.circuit.library import TwoLocal


H2_op = (-1.052373245772859 * I ^ I) + \
        (0.39793742484318045 * I ^ Z) + \
        (-0.39793742484318045 * Z ^ I) + \
        (-0.01128010425623538 * Z ^ Z) + \
        (0.18093119978423156 * X ^ X)


seed = 50
algorithm_globals.random_seed = seed
backend_cpu = AerSimulator(method = 'statevector')
backend_gpu = AerSimulator(method = 'statevector',device = 'GPU')
qi_cpu = QuantumInstance(backend_cpu, seed_transpiler=seed, seed_simulator=seed)
qi_gpu = QuantumInstance(backend_gpu, seed_transpiler=seed, seed_simulator=seed)

ansatz = TwoLocal(rotation_blocks='ry', entanglement_blocks='cz')
optimizer = SLSQP(maxiter=1000)
vqe_cpu = VQE(ansatz, optimizer=optimizer, quantum_instance=qi_cpu)
vqe_gpu = VQE(ansatz, optimizer=optimizer, quantum_instance=qi_gpu)
result_cpu = vqe_cpu.compute_minimum_eigenvalue(operator=H2_op)
result_gpu = vqe_gpu.compute_minimum_eigenvalue(operator=H2_op)

print("CPU : \n" ,result_cpu)
print("GPU : \n",result_gpu)