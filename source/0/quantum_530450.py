# https://github.com/KerileeCar/Quantum_Mechanics_Project/blob/55131cbddd5f3f0994f91a5f4d871f5e8a7d0aa2/Quantum.py
# ELEN4022 - Project - Group 01
# 09/05/2022
# Jesse Van Der Merwe (1829172)
# Keri-Lee Carstens (1384538)
# Tshegofatso Kale (1600916)

# - - - - - - - - - - IMPORTS - - - - - - - - - - #
import numpy as np
from matplotlib import pyplot as plt
from qiskit import *
from qiskit import QuantumRegister, QuantumCircuit, Aer, execute, IBMQ
from qiskit_finance.circuit.library import GaussianConditionalIndependenceModel as gaussian_conditional_independence_model
from qiskit.circuit.library import WeightedAdder, LinearAmplitudeFunction
from qiskit.utils import QuantumInstance
from qiskit.algorithms import IterativeAmplitudeEstimation, EstimationProblem

# - - - - - - - - - - QUANTUM CLASS - - - - - - - - - - #
class Quantum:
    def __init__(self, dp_1, dp_2, z_num, backend):
        """Parameterized constructor for the Quantum class."""
        self.backend_type = backend # Either 'Q' for quantum or 'C' for classical
        self.z_num_qubits = z_num # number of qubits used to represent the latent normal random variable Z
        self.z_trunc_value = 1 # min/maximum value used to truncate Z
        self.default_probability = [dp_1, dp_2] # the default probabilites for each of the assets
        self.k = len(self.default_probability) # the number of assets
        self.sensitivities = [0.1, 0.05] # the sensitivities of the default probabilities of each asset with respect to Z
        self.loss_given_default = [1, 2] # loss given default for each of the assets 
        self.circuit = gaussian_conditional_independence_model(self.z_num_qubits, self.z_trunc_value, self.default_probability, self.sensitivities)
        self.values_array, self.probs_array = self.get_probabilities_simulated()

    def set_data(self, dp_1, dp_2):
        """Sets the values in the list of default probabilities of the two assets."""
        self.default_probability = [dp_1, dp_2]

    def get_probabilities_simulated(self):
        job = execute(self.circuit, backend=Aer.get_backend("statevector_simulator"))
        default_p = np.zeros(self.k)

        values = []
        probs = []

        num_qubits = self.circuit.num_qubits
        state_vector = job.result().get_statevector().data

        for i in range(0, len(state_vector)):
            p = np.abs(state_vector[i]) ** 2
            losses = 0

            for j in range(self.k): # CLIENT 0 -> b[1] || CLIENT 1 -> b[0]       
                binary_string = '0' + str(num_qubits) + 'b'
                b = format(i, binary_string) # 000 000 - 001 001 - 010 010...
                if b[self.k-j-1] == "1":
                    default_p[j] = default_p[j] + p
                    losses = losses + self.loss_given_default[j]

            values.append(losses)
            probs.append(p)
            
        values_array = np.asarray(values)
        probs_array = np.asarray(probs)

        values_sorted = np.sort(np.unique(values_array))
        pdf = np.zeros(len(values_sorted))

        for i in range(0, len(values_sorted)):
            pdf[i] = pdf[i] + sum(probs_array[values_array == values_sorted[i]])
                
        return values_sorted, pdf

    def get_total_expected_loss_exact(self):
        """Calculates and returns the exact expected loss by performing the dot product between the values and probabilities."""
        self.values_array, self.probs_array = self.get_probabilities_simulated()
        self.exact_expected_loss = np.dot(self.values_array, self.probs_array)
        return self.exact_expected_loss

    def get_total_expected_loss_estimated(self):
        weighted_adder = WeightedAdder(self.z_num_qubits + self.k, [0] * self.z_num_qubits + self.loss_given_default)

        linear_objective = LinearAmplitudeFunction(
            weighted_adder.num_sum_qubits, 
            slope = [1],
            offset = [0],
            domain = (0, 2 ** weighted_adder.num_sum_qubits - 1),
            image = (0, sum(self.loss_given_default)),
            rescaling_factor = 0.25, # CHECK THIS VALUE - perhaps in the paper [Woerner2019]
            breakpoints = [0],
        )

        quantum_state = QuantumRegister(self.circuit.num_qubits, "state")
        quantum_objective = QuantumRegister(1, "objective")
        quantum_sum = QuantumRegister(weighted_adder.num_sum_qubits, "sum")
        quantum_carry = QuantumRegister(weighted_adder.num_carry_qubits, "carry")

        state_circuit = QuantumCircuit(quantum_state, quantum_objective, quantum_sum, quantum_carry) # Make the circuit
        state_circuit.append(self.circuit.to_gate(), quantum_state) # Append random variable
        state_circuit.append(weighted_adder.to_gate(), quantum_state[:] + quantum_sum[:]+ quantum_carry[:]) # Append aggregate circuit
        state_circuit.append(linear_objective.to_gate(), quantum_sum[:] + quantum_objective[:]) # Append linear objective function
        state_circuit.append(weighted_adder.to_gate().inverse(), quantum_state[:] + quantum_sum[:]+ quantum_carry[:]) # Append inverse of aggregate circuit
        state_circuit.draw()

        if(self.backend_type == "S"):
            quantum_instance = QuantumInstance(Aer.get_backend("aer_simulator"), shots=1000)
        elif(self.backend_type == "Q"):
            quantum_instance = QuantumInstance(self.get_IBMQ_backend(), shots=1000)

        estimation_problem = EstimationProblem(
            state_preparation = state_circuit,
            objective_qubits = [len(quantum_state)],
            post_processing = linear_objective.post_processing,
        )

        amplitude_estimation = IterativeAmplitudeEstimation(0.01, # EPSILON VALUE
            alpha = 0.05, # ALPHA VALUE
            quantum_instance = quantum_instance
        )

        result = amplitude_estimation.estimate(estimation_problem)
        confidence_interval = np.array(result.confidence_interval_processed) # Unused - useful for future improvements
        estimated_expected_loss = result.estimation_processed

        return estimated_expected_loss

    def get_IBMQ_backend(self):
        """Returns a backend object of the IBMQ provider from the perth quantum provider."""
        IBMQ.load_account()
        quantum_provider = IBMQ.get_provider(hub='ibm-q-education', group='uni-witwatersran-1', project='full-stack-2022')
        quantum_backend = quantum_provider.get_backend('ibm_perth')
        return quantum_backend