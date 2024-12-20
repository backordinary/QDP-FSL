# https://github.com/EXPmaster/QuantumErrorMitigation/blob/3f737db037dfd45962e056eb4fd457de893795ac/exp_vqe/cdr_trainer.py
import numpy as np
from qiskit import transpile
from qiskit.providers.aer import AerSimulator
from qiskit.providers.aer.noise import NoiseModel
import qiskit.providers.aer.noise as noise
from qiskit.quantum_info.operators import Operator, Pauli
from qiskit.quantum_info import Statevector
from qiskit.opflow import PauliOp

from mitiq import cdr
from mitiq.cdr import generate_training_circuits
from sklearn.linear_model import LinearRegression


class CDRTrainer:

    def __init__(self, noise_model):
        self.noisy_backend = AerSimulator(noise_model=noise_model)

    def fit(self, circuit, observable):
        circuit = transpile(circuit, basis_gates=['h', 's', 'rz', 'cx'])
        training_circuits = generate_training_circuits(
            circuit,
            num_training_circuits=60,
            fraction_non_clifford=0.1,
        )
        ideal_results = []
        noisy_results = []
        for circuit in training_circuits:
            ideal_results.append(self._simulate_ideal(circuit, observable))
            noisy_results.append(self._simulate_noisy(circuit, observable))
        ideal_results = np.array(ideal_results).reshape(-1, 1)
        noisy_results = np.array(noisy_results).reshape(-1, 1)
        self.reg = LinearRegression()
        self.reg.fit(noisy_results, ideal_results)
    
    def predict(self, noisy_data):
        return self.reg.predict(noisy_data)[0][0]

    def _simulate_ideal(self, circuit, observable):
        state_vector = Statevector(circuit)
        return state_vector.expectation_value(observable).real
        
    def _simulate_noisy(self, circuit, observable):
        tcircuit = circuit.copy()
        tcircuit.save_density_matrix()
        noisy_result = self.noisy_backend.run(transpile(tcircuit, self.noisy_backend)).result()
        density_matrix = noisy_result.data()['density_matrix']
        return density_matrix.expectation_value(observable).real


if __name__ == '__main__':
    from circuit_lib import random_circuit
    circuit = random_circuit(6, 2, 2)

    noise_model = NoiseModel()
    error_1 = noise.depolarizing_error(0.01, 1)  # single qubit gates
    error_2 = noise.depolarizing_error(0.01, 2)
    noise_model.add_all_qubit_quantum_error(error_1, ['u1', 'u2', 'u3', 'rx', 'ry', 'rz', 'i', 'x', 'y', 'z', 'h', 's', 't', 'sdg', 'tdg'])
    noise_model.add_all_qubit_quantum_error(error_2, ['cx', 'cy', 'cz', 'ch', 'crz', 'swap', 'cu1', 'cu3', 'rzz'])

    obs = PauliOp(Pauli('ZZIIII'))
    cdr_t = CDRTrainer(circuit, obs, noise_model)
    cdr_t.fit()
    noisy_data = cdr_t.simulate_noisy(cdr_t.circuit)
    noisy_data = np.array(noisy_data).reshape(-1, 1)
    print(noisy_data.shape)
    print(cdr_t.predict(noisy_data))