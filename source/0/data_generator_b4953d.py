# https://github.com/axelschacher/quantum_error_mitigation/blob/c67fc397d568a6d90b88ac76b26f1c73be78a975/quantum_error_mitigation/data/training_data/Data_Generator.py
import os
import random
# import matplotlib
# matplotlib.use('macosx') # TkAgg or agg for linux

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile, IBMQ
from qiskit.providers.aer import AerSimulator
from typing import List, Tuple, Any, Set, Optional

from quantum_error_mitigation.data.data_persistence.Data_Persistence import save_object, load_object
from quantum_error_mitigation.data.information_handler import Information_Handler


class Data_Generator():
    def __init__(self, n_qubits: int, n_samples: int, n_different_measurements: int, method: str, backend: str, path: Optional[str]=""):
        """
        Constructor of class Data_Generator.
        """
        self.provider = IBMQ.load_account()
        self.backend_name = backend
        if backend == "ibmq_quito":
            self.backend = self.provider.backend.ibmq_quito
        elif backend == "AerSimulatorQuito":
            self.simulated_backend = self.provider.backend.ibmq_quito
            self.backend = AerSimulator.from_backend(self.simulated_backend)
        elif backend == "NoiseModelAuckland":
            noise_model = load_object("../noise_model/noisemodel_only_measurement_errors.pkl")
            self.backend = AerSimulator(noise_model=noise_model)
        elif backend == "AerNairobi":
            self.simulated_backend = self.provider.backend.ibm_nairobi
            self.backend = AerSimulator.from_backend(self.simulated_backend)
        else:
            raise ValueError(f"Backend {backend} not supported. Choose between 'ibmq_quito', 'AerSimulatorQuito', 'NoiseModelAuckland' and 'AerNairobi'.")
        self.n_qubits = n_qubits
        self.n_different_measurements = n_different_measurements
        self.n_samples = n_samples
        self.method = method  # either rotation_angles or calibration_bits
        self.path = os.path.join(path)
        self.draw_circuits = False
        self.shots = 1024

    def generate_training_data(self, save_result=True):
        """
        Runs the whole data generation process for a data generator object.
        Parameters are defined on the data generator object.
        If available, circuits and metadata are loaded from the respective data path.
        Results are stored permanently to be used as long as data is current.
        """
        try:
            if self.method == "calibration_bits":
                self.circs = load_object(os.path.join(self.path, "circuits_calibraion_circuits.pkl"))
                self.information_for_training_data = load_object(os.path.join(self.path, "calibration_bits_list.pkl"))
            elif self.method == "rotation_angles":
                self.circs = load_object(os.path.join(self.path, "circuits_rotation_angles.pkl"))
                self.information_for_training_data = load_object(os.path.join(self.path, "rotation_angles.pkl"))
        except FileNotFoundError:
            if self.method == "calibration_bits":
                self.circs, self.information_for_training_data = self.create_calibration_circuits()
            elif self.method == "rotation_angles":
                self.circs, self.information_for_training_data = self.create_random_qubit_rotation_circuits()
            else:
                raise ValueError(f"Method {self.method} not supported. Choose between 'calibration_bits' and 'rotation_angles'.")
        self.transpiled_qc = self.transpile_to_backend()
        job = self.run_on_backend()
        result = self.get_job_result(job)
        if save_result:
            if self.method == "calibration_bits":
                samples, desired_solutions = self.save_calibration_bits_samples_to_numpy()
            elif self.method == "rotation_angles":
                samples, desired_solutions = self.save_rotation_samples_to_numpy()
        else:
            return result

    def create_calibration_circuits(self) -> Tuple[List[Any], List[str]]:
        """
        Generate randomly number_of_circuits-many different calibration circuits out of the 2ˆnumber_of_qubits possible states on a specific machine.
        """
        circs = []
        calibration_bits_list = self._generate_random_calibration_bits(n_qubits=self.n_qubits, number_of_samples=self.n_samples)
        for elem in calibration_bits_list:
            qc = self._calibration_bits_to_circuit(elem)
            circs.append(qc)
        save_object(circs, os.path.join(self.path, "circuits_calibraion_circuits.pkl"))
        save_object(calibration_bits_list, os.path.join(self.path, "calibration_bits_list.pkl"))
        return circs, calibration_bits_list

    def create_random_qubit_rotation_circuits(self) -> Tuple[List[Any], List[str]]:
        """
        Generate randomly number_of_circuits-many different circuits that rotate each qubit by a random angle and measure it in the end.
        """
        circs = []
        rotation_angles = np.random.uniform(low=0., high=2*np.pi, size=(self.n_samples, self.n_qubits))
        for i in range(rotation_angles.shape[0]):
            qc = self._rotation_angles_to_circuit(rotation_angles[i, :])
            circs.append(qc)
        save_object(circs, os.path.join(self.path, "circuits_rotation_angles.pkl"))
        save_object(rotation_angles, os.path.join(self.path, "rotation_angles.pkl"))
        return circs, rotation_angles

    def initialize_quantum_circuit(self) -> Tuple[Any, Any, Any]:
        """
        Initializes a Qiskit Quantum circuit with n_qubits qubits and the same number of classical bits.

        Returns:
            qc: The quantum circuit object.
            qreg_q: The quantum register of qc.
            creg_c: The classical register of qc.
        """
        qreg_q = QuantumRegister(self.n_qubits, 'q')
        creg_c = ClassicalRegister(self.n_qubits, 'c')
        qc = QuantumCircuit(qreg_q, creg_c) # qubits are automatically set to zero
        return qc, qreg_q, creg_c

    def transpile_to_backend(self) -> Any:
        """
        Transpiles all circuits in self.circs to the backend stored in the data generator object.
        If desired, draws and saves circuit plots to the circuit plot folder.
        """
        transpiled = transpile(self.circs, backend=self.backend)
        if self.draw_circuits:
            style = {'fontsize': 16, 'subfontsize': 14}
            for i in range(len(self.circs)):
                transpiled[i].draw(output='mpl', style=style, filename=f'{self.method}_circuit_plots/circuit_{str(self.information_for_training_data[i])}_transpiled.png')
                plt.close()
        return transpiled

    def run_on_backend(self) -> Any:
        """
        Runs the transpled circuits on the backend.
        Stores the job_properties to download results from the IBM server after execution, if desired.
        """
        job = self.backend.run(self.transpiled_qc, shots=self.shots)
        self.job_id = job.job_id()
        # save job properties to be able to access the results later
        job_properties = {
                        "backend": self.backend,
                        "information_for_training_data": self.information_for_training_data,
                        "circs": self.circs,
                        "transpiled_qc": self.transpiled_qc,
                        "job_id": self.job_id,
                        "shots": self.shots
                        }
        save_object(job_properties, os.path.join(self.path, "job_properties.pkl"))
        return job

    def get_job_result(self, job=None) -> Any:
        """
        Loads and saves results from a job.

        Returns:
            result: qiskit.result object containing the job results.
        """
        self._load_job_properties_in_self()
        if job is None:
            job = self.backend.retrieve_job(self.job_id)
        result = job.result()
        save_object(result, os.path.join(self.path, f"result_{self.method}.pkl"))
        return result

    def save_calibration_bits_samples_to_numpy(self, result: Optional[Any] = None):
        """
        One row per sample, the first self.n_qubits values are the measured qubits, the next value is the measured probability for this state and the last one the integer value of this quantum register.
        Issue: Cannot store solutions for quantum registers in superposition that (only 1 outcome possible yet).

        Returns:
            samples: Array of shape [n_samples, n_different_measurements, n_qubits+2[bits+probability+integer_value]] containing the measured data.
            desired_solutions: Array of shape [n_samples, number_of_considered_states, n_qubits+2[bits, probability, integer_value]] containing the states of the quantum register that we expect and their theoretical probability.
        """
        if result is None:
            result = load_object(os.path.join(self.path, "result_calibration_bits.pkl"))
        samples = np.zeros((self.n_samples, self.n_different_measurements, self.n_qubits+2))
        desired_solutions = np.zeros((self.n_samples, 1, self.n_qubits+2)) # extra dimension for cases where we have more than one state with probability > 0
        for i, sample in enumerate(self.information_for_training_data):
            big_endian_calibration_bits = sample[::-1] # returned strings are little-endian, i.e. creg_c[0] is the right-most bit/character in result.get_counts(), so we reverse the loop order.
            for bit_number, bit in enumerate(big_endian_calibration_bits):
                desired_solutions[i, 0, bit_number] = bit
            desired_solutions[i, 0, self.n_qubits] = 1.
            desired_solutions[i, 0, self.n_qubits+1] = Information_Handler.quantum_register_to_integer(big_endian_calibration_bits, big_endian=True)
            samples = self._process_and_store_sample_to_samples(result, i, samples)
        save_object(samples, os.path.join(self.path, "training_inputs.pkl"))
        save_object(desired_solutions, os.path.join(self.path, "training_solutions.pkl"))
        return samples, desired_solutions

    def save_rotation_samples_to_numpy(self, result: Optional[Any] = None):
        """
        One row per sample, the first self.n_qubits values are the measured qubits, the next value is the measured probability for this state and the last one the integer value of this quantum register.

        Returns:
            samples: Array of shape [n_samples, n_different_measurements, n_qubits+2[bits+result_counts+integer_value]] containing the measured data.
            desired_solutions: Array of shape [n_samples, number_of_considered_states, n_qubits+2[bits, probability, integer_value]] containing the states of the quantum register that we expect and their theoretical probability.
        """
        if result is None:
            result = load_object(os.path.join(self.path, "result_rotation_angles.pkl"))
        samples = np.zeros((self.n_samples, self.n_different_measurements, self.n_qubits+2)) # [sample (set of rotation angles for 1 quantum register), measured outcomes, [bits, probability, integer_value_of_state]]
        number_of_considered_states = self.n_different_measurements  # or 2**self.n_qubits
        assert number_of_considered_states <= 2**self.n_qubits
        desired_solutions = np.zeros((self.n_samples, number_of_considered_states, self.n_qubits+2)) # [sample, theoretical states, [bits, theoretical probability, integer_value_of_state]
        for sample in range(self.information_for_training_data.shape[0]):
            input_angles = self.information_for_training_data[sample, :]
            probabilities = np.zeros((2**self.n_qubits))
            states = np.zeros((2**self.n_qubits, self.n_qubits))
            for j in range(2**self.n_qubits): # compute theoretical probability of each state
                state = Information_Handler.integer_value_to_classical_register(j, n_bits=self.n_qubits, big_endian=True)
                states[j, :] = state
                probabilities[j] = self._compute_ry_rotation_theoretical_probability(state, input_angles)
            # get those with the highest probability
            k_highest_probability_states = np.flip(np.argsort(probabilities)[-number_of_considered_states:])
            desired_solutions[sample, :, 0:self.n_qubits] = states[k_highest_probability_states, :]
            desired_solutions[sample, :, self.n_qubits] = probabilities[k_highest_probability_states]
            desired_solutions[sample, :, self.n_qubits+1] = k_highest_probability_states
            samples = self._process_and_store_sample_to_samples(result, sample, samples)
        save_object(samples, os.path.join(self.path, "training_inputs.pkl"))
        save_object(desired_solutions, os.path.join(self.path, "training_solutions.pkl"))
        return samples, desired_solutions

    def print_qubit_properties_single_readout_errors(self) -> None:
        """
        Prints qubit readout error rates for each qubit of the backend.
        """
        properties = self.backend.properties().to_dict()
        print(f"qubit properties (single readout errors):")
        for qubit in range(self.backend.configuration().n_qubits): # iterate over all available qubits
            print("qubit #", qubit)
            prob_meas0_prep1 = properties["qubits"][qubit][5]["value"]
            prob_meas1_prep0 = properties["qubits"][qubit][6]["value"]
            print(f"prob_meas0_prep1 = {prob_meas0_prep1:.3f}")
            print(f"prob_meas1_prep0 = {prob_meas1_prep0:.3f}")

    def _generate_random_calibration_bits(self, n_qubits: int, number_of_samples: int) -> List[List[int]]:
        """
        Generates the bitstrings for number_of_samples randomly chosen, different bitstrings.

        Args:
            n_qubits: int: Number of qubits in the quantum circuit.
            number_of_samples: int: Number of different bitstrings to create. Has to be less than or equal to 2**n_qubits, because no more different states exist.
        Returns:
            bitstrings_as_list: list[list[int]] containing the bitstrings.
        """
        assert number_of_samples <= 2**n_qubits
        bitstring_set: Set[str] = set() # We use a set to avoid duplicates.
        format = '0' + str(n_qubits) + 'b'
        while (len(bitstring_set) < number_of_samples):
            bitstring_set.add(f'{random.getrandbits(n_qubits):={format}}')
        bitstrings = list(bitstring_set) # convert to list to get unique idx of each bitstring
        # convert strings to arrays
        bitstrings_as_list = [[int(bit) for bit in list(bitstring)] for bitstring in bitstrings]
        return bitstrings_as_list

    def _calibration_bits_to_circuit(self, calibration_bits: List[int]) -> Any:
        """
        Converts the list of calibration bitstrings to the respective quantum circuit.

        Args:
            calibration_bits: list[int]: List of the bitstrings to create the circuits from.

        Returns:
            qc: the quantum circuit corresponding with the ideal readout result of calibration_bits.
        """
        qc, qreg_q, creg_c = self.initialize_quantum_circuit()
        i=0
        for bit in calibration_bits:
            if bit==1:
                qc.x(qreg_q[len(calibration_bits)-i-1]) # the first bit is the highest value one. So we need to go backwards through the register and subtract 1 to get a zero-index base.
            elif bit !=0:
                raise ValueError(f'Invalid bit {bit}. Bits have to be 0 or 1.')
            i+=1
        qc.barrier(qreg_q)
        qc.measure(qreg_q, creg_c)
        if self.draw_circuits:
            style = {'fontsize': 16, 'subfontsize': 11}
            qc.draw(output='mpl', style=style, filename=f'{self.method}_circuit_plots/circuit_{calibration_bits}_original.png')
            plt.close()
        return qc

    def _rotation_angles_to_circuit(self, rotation_angles: npt.NDArray):
        """
        Generates a quantum circuits that implements the Ry-Rotation by the given angle to the corresponding qubit.
        Plots the circuits to the circuit_plots directory, if desired

        Args:
            rotation_angles: np.NDArray: array of the rotation angle for each qubit.

        Returns:
            qc: qiskit.QuantumCircuit: the circuit implementing the given qubit rotations.
        """
        qc, qreg_q, creg_c = self.initialize_quantum_circuit()
        for i, angle in enumerate(rotation_angles):
            qc.ry(theta=angle, qubit=qreg_q[i])
        qc.barrier(qreg_q)
        qc.measure(qreg_q, creg_c)
        if self.draw_circuits:
            style = {'fontsize': 16, 'subfontsize': 11}
            qc.draw(output='mpl', style=style, filename=f'{self.method}_circuit_plots/circuit_{rotation_angles[0]}_original.png')
            plt.close()
        return qc

    def _compute_ry_rotation_theoretical_probability(self, state: List[int], rotation_angles: npt.NDArray):
        """
        Computation of theoretical probability to measure a given state based on the given rotation angles implemented from Kim2021.
        """
        # index 0 is qubit 0. Therefore input strings should be big-endian to traverse in given order.
        assert len(state) == rotation_angles.shape[0]
        probability = 1.
        for i in range(rotation_angles.shape[0]):
            theta_i = rotation_angles[i]
            bit_i = state[i]
            cos_part = np.cos(theta_i/2) ** (1-bit_i)
            sin_part = np.sin(theta_i/2) ** bit_i
            probability = probability * cos_part * sin_part
        probability = np.abs(probability) ** 2
        return probability

    def _load_job_properties_in_self(self) -> None:
        """
        Loads the job properties into the data generator object.
        Needed for later access to an executed job.
        """
        job_properties = load_object(os.path.join(self.path, "job_properties.pkl"))
        self.backend = job_properties["backend"]
        self.information_for_training_data = job_properties["information_for_training_data"]
        self.circs = job_properties["circs"]
        self.transpiled_qc = job_properties["transpiled_qc"]
        self.job_id = job_properties["job_id"]

    def _process_and_store_sample_to_samples(self, result: Any, sample: int, samples: npt.NDArray) -> npt.NDArray:
        """
        Stores a single sample (given by index) into the samples array.
        """
        result_counts = result.get_counts()[sample]
        # sort result_counts by probability
        sorted_result_counts = sorted(result_counts, key=result_counts.get, reverse=True)
        sorted_result_counts = sorted_result_counts[0:self.n_different_measurements]
        for j, measured_outcome in enumerate(sorted_result_counts):
            big_endian_measured_outcome = measured_outcome[::-1]
            measured_outcome_integer = Information_Handler.quantum_register_to_integer(big_endian_measured_outcome, big_endian=True)
            for k, bit in enumerate(big_endian_measured_outcome):
                samples[sample, j, k] = bit
            samples[sample, j, self.n_qubits] = result_counts[measured_outcome]/self.shots
            samples[sample, j, self.n_qubits+1] = measured_outcome_integer
        return samples


if __name__ == '__main__':
    """
    Creates the Data_Generator object with the properties and generates training data.
    """
    generator = Data_Generator(n_qubits=5, n_samples=100, n_different_measurements=32, method="rotation_angles", backend="ibmq_quito", path="./4000_samples_rotation_5_qubits")  # method either rotation_angles or calibration_bits; backend either ibmq_quito, AerSimulatorQuito or NoiseModelAuckland; method either rotation_angles or calibration_bits
    generator.generate_training_data()
