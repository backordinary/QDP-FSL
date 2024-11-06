# https://github.com/AthenaCaesura/HtN-Test-Circuit/blob/ae08d848c65d9a2af51beb6830c24c3e16bbf663/src/test_circuit.py
import qiskit.providers.aer.noise as noise
from orquestra.integrations.qiskit.simulator import QiskitSimulator
from orquestra.quantum.circuits import Circuit, H, X

error = noise.depolarizing_error(0.1, 1)

# declare noise model
noise_model = noise.NoiseModel()
noise_model.add_all_qubit_quantum_error(error, ["h"])

circ = Circuit([H(0), X(0), H(0)])

# declare backend and run circuit
qiskit_sim = QiskitSimulator(device_name="aer_simulator", noise_model=noise_model)
measurements = qiskit_sim.run_circuit_and_measure(circ, 1000)

# before sandwiching, should have some errors. So some results will be "01".
print(measurements.get_counts())

"""
from orquestra.quantum.backends import PauliSandwichBackend

sandwiched_qiskit_backend = PauliSandwichBackend(CNOT, None, qiskit_sim)
measurements = qiskit_sim.run_circuit_and_measure(circ, 1000)

# after sandwiching, we should have no errors. Result should be {"11": 1000}
# getting the dictionary {"111", 1000} indicates errors have been eliminated
print(measurements.get_counts())
"""
