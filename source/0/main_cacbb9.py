# https://github.com/larsesser/bachelor-code/blob/7331c81e40b9d4de63425e8c00fd23f72429c09d/main.py
import numpy as np
from qiskit import Aer
from qiskit.providers.aer import AerSimulator
from qiskit.providers.aer.backends.aerbackend import AerBackend
from qiskit.providers.aer.noise import NoiseModel
from qiskit.test.mock import FakeVigo

from noise import measure_errors
from ordered_operator import IGate, OrderedOperator, ZGate
from util import print_result
from w import w_matrix, w_matrix_inverse
from test_mitigation import benchmark_random

# The operator of interest. Additional gates may be added like
#  operator = OrderedOperator(ZGate(2), IGate(1), ZGate(0))
operator = OrderedOperator(ZGate(0))
print(f"Operator: {operator}")

# prepare the noisy simulator
noise_backend = FakeVigo()
noise_model = NoiseModel().from_backend(noise_backend, gate_error=False, thermal_relaxation=False)
noisy_sim: AerBackend = AerSimulator(noise_model=noise_model)
# Seed the simulator to make results reproducible. Remove for actual runs.
noisy_sim.set_options(seed_simulator=42)
print(f"Shots: {noisy_sim.options.get('shots')}")

# prepare the noiseless simulator
noiseless_sim: AerSimulator = Aer.get_backend('aer_simulator')
noiseless_sim.set_options(shots=noisy_sim.options.get("shots"))
# Seed the simulator to make results reproducible. Remove for actual runs.
noiseless_sim.set_options(seed_simulator=42)

# calculate (and cache) the w matrix and its inverse
error_probabilities = measure_errors(noisy_sim, operator.N)
w = w_matrix(operator.N, error_probabilities)
w_inverse = w_matrix_inverse(operator.N)

# set a seed for the random angles generation
np.random.seed(42)

print(f"Error probabilities: {error_probabilities}")
print(f"W matrix: {w}")
print(f"W inverse: {w_inverse}")

results = []
iterations = 20
for _ in range(iterations):
    result = benchmark_random(
        operator,
        noiseless_backend=noiseless_sim,
        noisy_backend=noisy_sim,
        error_probabilities=error_probabilities)
    # to avoid dividing by small numbers
    if abs(result.noiseless) < 0.01:
        continue
    results.append(result)

print(f"{len(results)} out of {iterations} results were taken into account.\n")
print_result(results)
