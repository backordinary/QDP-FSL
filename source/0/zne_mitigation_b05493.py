# https://github.com/aakif-akhtar/error_mitigation/blob/070f3ebc2a332b87b8fcdf5119a72dfa742a5d57/zne_mitigation.py
from qiskit import IBMQ
# from qiskit.providers.ibmq import least_busy

# from qiskit import QuantumCircuit, execute

from qiskit.providers.aer.noise import NoiseModel

from libraries.vqe_ansatz import vqe

import numpy as np

# from mitiq.zne.scaling import (
#     fold_gates_from_left,
#     fold_gates_from_right,
#     fold_global,
#     fold_all,
# )
from mitiq.zne.inference import LinearFactory, RichardsonFactory, PolyFactory
from mitiq.zne.zne import execute_with_zne

# from mitiq.zne.zne import execute_with_zne

from qiskit.opflow import StateFn, CircuitStateFn

from qiskit import Aer
# from qiskit.test.mock import FakeVigo
from qiskit.utils import QuantumInstance
from qiskit.opflow import CircuitSampler, StateFn, ExpectationFactory

from qiskit.algorithms.optimizers import COBYLA, SLSQP, SPSA, QNSPSA


IBMQ.save_account(
    "4d7226ec37d38370454cbd7111e98a765d1cc1e6de0fa93a28a7b5918b6dd4b5c91f9fe74c748e270d8da2b5ef7c451b6550df0973de468ec4965a26b1b5c348",
    overwrite=True,
)
provider = IBMQ.load_account()
provider = IBMQ.get_provider("ibm-q")
# device = least_busy(provider.backends(filters=lambda x: x.configuration().n_qubits >= 3 and
#                                    not x.configuration().simulator and x.status().operational==True))
# print("Running on current least busy device: ", device)


# Build noise model from backend properties
# provider = IBMQ.load_account()
# backend = provider.get_backend(str(device))
backend = provider.get_backend("ibm_nairobi")

noise_model = NoiseModel.from_backend(backend)

# Get coupling map from backend
coupling_map = backend.configuration().coupling_map

# Get basis gates from noise model
# basis_gates = noise_model.basis_gates

# method: scale noise by 1 and 2, then extrapolate linearly to the zero noise limit.
# linear_fac = LinearFactory(scale_factors=[1.0, 2.0])

# method: scale noise by 1, 2 and 3, then evaluate the Richardson extrapolation.
# richardson_fac = RichardsonFactory(scale_factors=[1.0, 2.0, 3.0])
richardson_fac = RichardsonFactory(scale_factors=[1.0, 1.5, 2.0, 2.5])

# method: scale noise by 1, 2, 3, and 4, then extrapolate quadratically to the zero noise limit.
# poly_fac = PolyFactory(scale_factors=[1.0, 2.0, 3.0, 4.0], order=2)


def objective_function_zne(params):
    # qubit_op = vqe(ansatz_id=3).create_hamiltonian()

    def executor(ansatz):
        
        qubit_op = vqe(ansatz_id=2).create_hamiltonian()

        # creating the wavefunction from the circuit
        psi = CircuitStateFn(ansatz)

        # define your backend or quantum instance (where you can add settings)
        # backend = provider.get_backend(str(device))
        backend = Aer.get_backend("qasm_simulator")
        # q_instance = QuantumInstance(backend, shots=1024)
        q_instance = QuantumInstance(
            backend, shots=1024, noise_model=noise_model, coupling_map=coupling_map
        )

        # define the state to sample

        # convert to expectation value
        expectation = ExpectationFactory.build(operator=qubit_op, backend=q_instance)
        measurable_expression = expectation.convert(
            StateFn(qubit_op, is_measurement=True)
        )
        expect_op = measurable_expression.compose(psi).reduce()
        # get state sampler (you can also pass the backend directly)
        sampled_expect_op = CircuitSampler(q_instance).convert(
            expect_op
        )
        energy_evaluation = np.real(sampled_expect_op.eval())

        # evaluate
        # print('Sampled:', sampler.eval().real)

        return energy_evaluation

    ansatze = vqe(ansatz_id=2).get_circ_2(params=params)
    # expectation_folded = executor(ansatze)
    # print("ansatz:", ansatze)
    zne_expval = execute_with_zne(ansatze, executor, factory=richardson_fac)
    print("mitigated expectation:", zne_expval)
    return zne_expval


if __name__ == "__main__":
    np.random.seed(0)
    # optimizer = SPSA(maxiter=5, callback=vqe_callback)
    optimizer = COBYLA(maxiter=120)

    num_vars = vqe(ansatz_id=2).get_ansatz().num_parameters
    print("num_vars:",num_vars)
    params = np.random.randn(num_vars)

    mit_out = optimizer.minimize(fun=objective_function_zne, x0=params)
    # mit_out = optimizer.optimize(num_vars, objective_function=objective_function_zne, initial_point=params)

    print("iteration %d mitigated results is : " % (0 + 1), mit_out)
    np.random.seed(0)
    # optimizer = SPSA(maxiter=5, callback=vqe_callback)
    optimizer = COBYLA(maxiter=120)

    num_vars = vqe(ansatz_id=2).get_ansatz().num_parameters
    print("num_vars:",num_vars)
    params = np.random.randn(num_vars)

    mit_out = optimizer.minimize(fun=objective_function_zne, x0=params)
    # mit_out = optimizer.optimize(num_vars, objective_function=objective_function_zne, initial_point=params)

    print("iteration %d mitigated results is : " % (1 + 1), mit_out)
    np.random.seed(0)
    # optimizer = SPSA(maxiter=5, callback=vqe_callback)
    optimizer = COBYLA(maxiter=120)

    num_vars = vqe(ansatz_id=2).get_ansatz().num_parameters
    print("num_vars:",num_vars)
    params = np.random.randn(num_vars)

    mit_out = optimizer.minimize(fun=objective_function_zne, x0=params)
    # mit_out = optimizer.optimize(num_vars, objective_function=objective_function_zne, initial_point=params)

    print("iteration %d mitigated results is : " % (2 + 1), mit_out)
    np.random.seed(0)
    # optimizer = SPSA(maxiter=5, callback=vqe_callback)
    optimizer = COBYLA(maxiter=120)

    num_vars = vqe(ansatz_id=2).get_ansatz().num_parameters
    print("num_vars:",num_vars)
    params = np.random.randn(num_vars)

    mit_out = optimizer.minimize(fun=objective_function_zne, x0=params)
    # mit_out = optimizer.optimize(num_vars, objective_function=objective_function_zne, initial_point=params)

    print("iteration %d mitigated results is : " % (3 + 1), mit_out)
    np.random.seed(0)
    # optimizer = SPSA(maxiter=5, callback=vqe_callback)
    optimizer = COBYLA(maxiter=120)

    num_vars = vqe(ansatz_id=2).get_ansatz().num_parameters
    print("num_vars:",num_vars)
    params = np.random.randn(num_vars)

    mit_out = optimizer.minimize(fun=objective_function_zne, x0=params)
    # mit_out = optimizer.optimize(num_vars, objective_function=objective_function_zne, initial_point=params)

    print("iteration %d mitigated results is : " % (4 + 1), mit_out)
