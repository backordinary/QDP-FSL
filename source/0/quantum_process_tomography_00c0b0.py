# https://github.com/joostvdzalm/Internation-QI-team/blob/ea828091888f8b3a04f818b210297fa587541e93/Quantum%20Process%20Tomography/Quantum_Process_Tomography.py
import matplotlib.pyplot as plt  # for plotting
import matplotlib.cm as cm  # for colour maps
from qiskit import Aer  # for simulating on own pc
import numpy as np  # for arrays
import qiskit as qk  # for quantum
import math  # for simple math
from progress.bar import IncrementalBar  # so you can see the pc is doing something
# So you can see the progress with the bar run this file in the Terminal by typing: "python main.py"


class QuantumProcessTomography:
    """
    This class contains all functions necessary for doing n-dimensional Qubit Process Tomography.
    The process you want to do tomography on can be defined in the 'quantum_process()' function.
    The calculation size grows dramatically as the number of qubits increases. So make sure the
    dimensionality of the tomography is the same as that of the process. So that no computational
    resources are being wasted and you can have the results ASAP.
    """
    counter = 0

    def __init__(self, number_of_qubits, number_of_shots=8192):
        """
        :param number_of_qubits: The amount of qubits involved in the Quantum Process you want to do QPT on.
        :param number_of_shots: The number of shots to do for every individual experiment, default is 8192.

        In this function the tools that are being used throughout the class are being initialized.
        """
        self.n = number_of_qubits
        self.backend = Aer.get_backend('qasm_simulator')
        self.shots = number_of_shots

        # Setting up some tools which are used for the calculations
        self.cardinal_states = [np.array([1, 0]),
                                np.array([0, 1]),
                                1/np.sqrt(2) * np.array([1, 1]),
                                1/np.sqrt(2) * np.array([1, -1]),
                                1/np.sqrt(2) * np.array([1, complex(0, 1)]),
                                1/np.sqrt(2) * np.array([1, complex(0, -1)])]
        self.directions = ['I', 'X', 'Y', 'Z']
        self.pauli_vector_string = self.compute_pauli_vector_string()  # Contains the combined direction labels

        # Creating initial arrays so they are ready for computations
        self.qubit_inputs = np.zeros((len(self.cardinal_states) ** self.n, self.n, 2), dtype=np.complex_)
        for state_index in range(np.shape(self.qubit_inputs)[0]):
            for qubit_index in range(np.shape(self.qubit_inputs)[1]):
                self.qubit_inputs[state_index, qubit_index, :] = np.array([0, 0])  # Init. with an empty bloch vector

        self.pauli_input_matrix = np.zeros((len(self.directions) ** self.n, len(self.cardinal_states) ** self.n))
        self.pauli_output_matrix = np.zeros((len(self.directions) ** self.n, len(self.cardinal_states) ** self.n))
        self.theoretical_output_matrix = np.zeros((len(self.directions) ** self.n, len(self.cardinal_states) ** self.n))
        self.pauli_transfer_matrix = np.zeros((4 ** self.n, 4 ** self.n))
        self.theoretical_transfer_matrix = np.zeros((4 ** self.n, 4 ** self.n))

        # Initializing the circuit
        self.qubits = qk.QuantumRegister(self.n)
        self.classical_bits = qk.ClassicalRegister(self.n)
        self.circuit = qk.QuantumCircuit(self.qubits, self.classical_bits)  # Changes per experiment

    def compute_pauli_vector_string(self):
        """
        This function is called on initialization, it creates a list which contains the combined directional labels of
        the Pauli vectors. The second for loop works as follows, for every index it tries to find a character at a
        position which it can use, if it cannot find any an 'I' will be used. By looping through all indices in this way
        all combinations of directions are made. This is done similarly to a number system (binary, decimal,
        hexadecimal). With here the I, X, Y, Z representing the characters to form 'numbers'.

        :return: combined_directions
        """
        length = len(self.directions) ** self.n
        combined_directions = []
        start_string = ''
        for i in range(self.n):  # Creating a string "n * I"
            start_string += 'I'

        for index in range(length):  # This for loop creates strings with all possible directions
            leftover_index = index
            new_string_list = list(start_string)
            for position in reversed(range(self.n)):
                if leftover_index >= (3 * 4**position):
                    new_string_list[-1-position] = 'Z'
                    leftover_index -= 3 * 4**position
                elif leftover_index >= (2 * 4**position):
                    new_string_list[-1 - position] = 'Y'
                    leftover_index -= 2 * 4 ** position
                elif leftover_index >= (4**position):
                    new_string_list[-1 - position] = 'X'
                    leftover_index -= 4**position
                elif leftover_index <= 0:
                    break
            new_string = "".join(new_string_list)
            combined_directions.append(new_string)

        return combined_directions

    def get_pauli_matrix(self, direction):
        """
        This function simply returns the Pauli matrices needed depending on the direction given in the input.
        :param direction: The direction you want the pauli-matrix for.
        :return: pauli_matrix, returns either Identity or Pauli X/Y/Z
        """
        pauli_matrix = np.zeros((2, 2))
        if direction == 'I':
            pauli_matrix = np.array([[1, 0], [0, 1]])
        elif direction == 'X':
            pauli_matrix = np.array([[0, 1], [1, 0]])
        elif direction == 'Y':
            pauli_matrix = np.array([[0, complex(0, -1)], [complex(0, 1), 0]])
        elif direction == 'Z':
            pauli_matrix = np.array([[1, 0], [0, -1]])
        return pauli_matrix

    def compute_pauli_input_vector(self, qubit_states):
        """
        In this function the Pauli state vector is calculated using the kron/tensor product.
        :param qubit_states: The input can be characterized by the combined states of the qubits.
        :return: pauli_input_vector, this is the Pauli state vector of the input.
        """
        pauli_input_vector = np.zeros((len(self.directions) ** self.n, 1))

        input_state_vector = qubit_states[0]  # Pick the first element
        for state in qubit_states[1:]:  # calculate the kron product for all states combined
            input_state_vector = np.kron(state, input_state_vector)

        index = 0
        for combined_directions in self.pauli_vector_string:
            directional_matrix = self.get_pauli_matrix(list(combined_directions)[0])  # Pick the first element
            for char in list(combined_directions)[1:]:  # calc. the kron prod. for all possible directional matrices
                directional_matrix = np.kron(self.get_pauli_matrix(char), directional_matrix)

            pauli_input_vector[index] = np.real(np.dot(np.conjugate(input_state_vector), np.dot(directional_matrix, input_state_vector)))
            index += 1

        return pauli_input_vector[:, 0]

    def compute_all_qubit_inputs(self):
        """
        This function computes an array containing all possible qubit inputs. Given that we have n qubits and that we
        want all possible combinations of cardinal states. The operation used to calculate this list is similar to what
        was used in compute_pauli_vector_string(). The second for loop works as follows, for every index it tries to
        find a cardinal state at a position which it can use, if it cannot find any an '0 state' will be used (notice
        that the start_input is first initialized with these states in the first for loop.  By looping through all
        indices in this way all combinations of directions are made.
        :return: qubit_inputs, the list of all combinations of cardinal states for the n qubits.
        """
        qubit_inputs = self.qubit_inputs

        start_input = []

        for i in range(self.n):  # Creating a list "n * np.array([1, 0])"
            start_input.append(np.array([1, 0]))

        length = len(self.cardinal_states) ** self.n
        for state_index in range(length):  # This for loop goes over all possible inputs
            leftover_index = state_index
            new_input = start_input
            for position in reversed(range(self.n)):
                if leftover_index >= (5 * 6**position):
                    new_input[position] = self.cardinal_states[5]
                    leftover_index -= 5 * 6**position
                elif leftover_index >= (4 * 6**position):
                    new_input[position] = self.cardinal_states[4]
                    leftover_index -= 4 * 6**position
                elif leftover_index >= (3 * 6**position):
                    new_input[position] = self.cardinal_states[3]
                    leftover_index -= 3 * 6**position
                elif leftover_index >= (2 * 6**position):
                    new_input[position] = self.cardinal_states[2]
                    leftover_index -= 2 * 6**position
                elif leftover_index >= (6**position):
                    new_input[position] = self.cardinal_states[1]
                    leftover_index -= 6**position
                elif leftover_index <= 0:
                    break
            qubit_inputs[state_index, :, :] = new_input

        self.qubit_inputs = qubit_inputs
        return qubit_inputs

    def compute_pauli_input_matrix(self):
        """
        This function simply returns an array containing all possible input Pauli State vectors. It does this by calling
        the compute_pauli_input_vector() on every input which was found with compute_all_qubit_inputs().
        :return: pauli_input_matrix
        """
        pauli_input_matrix = self.pauli_input_matrix
        self.compute_all_qubit_inputs()

        state_index = 0
        for new_input in self.qubit_inputs:
            pauli_input_matrix[:, state_index] = self.compute_pauli_input_vector(new_input)
            state_index += 1

        self.pauli_input_matrix = pauli_input_matrix
        return pauli_input_matrix

    def clear_circuit(self):
        """
        This function initializes the circuit and bits again to clear out the old circuit and bits.
        :return: none
        """
        self.qubits = qk.QuantumRegister(self.n)
        self.classical_bits = qk.ClassicalRegister(self.n)
        self.circuit = qk.QuantumCircuit(self.qubits, self.classical_bits)

    def setup_circuit(self, qubit_states):
        """
        This function does some initialization for the input states that are given.
        :param qubit_states: The input states for the QPT.
        :return: none
        """
        qubit_index = 0
        for state in qubit_states:
            self.circuit.initialize(state, qubit_index)  # Initialize the i-th qubit using a complex vector
            qubit_index += 1

    def quantum_process(self):
        """
        This function is used to do the actual quantum process that we are doing QPT on. CHANGE THIS AS YOU SEE FIT. You
        can do QPT on any process as you wish. Make sure that the dimension of the process defined here is the same as
        that when you create an instance of this QubitProcessTomography class.
        :return: none
        """
        for i in range(self.n):
            self.circuit.h(self.qubits[i])  # Do a Hadadamard gate on every input

    def compute_pauli_output_vector(self, qubit_states):
        """
        I would say this function is at the core of the class. In this function the backend is actually used and the
        experiments are being done. From the backend we get a histogram containing the states and amount of counts for
        each specific state. Based on the measurements for X, Y and Z and the non-measurement I we can then calculate
        the expected values in the Pauli state vector.
        :param qubit_states: The input qubit state(s) which are used to setup the circuits and do the experiments.
        :return: pauli_output_vector, this is the Pauli state vector of the output.
        """
        pauli_output_vector = np.zeros((len(self.directions) ** self.n, 1))
        index = 0
        for combined_directions in self.pauli_vector_string:
            # Start with clearing out the old circuit, setting up and adding the process
            self.clear_circuit()
            self.setup_circuit(qubit_states)
            self.quantum_process()

            # These lists are used so that the qubits with direction 'I' are not measured
            measured_qubits = []
            measured_bits = []

            for i in range(len(combined_directions)):
                if combined_directions[i] == 'X':  # Do an X measurement
                    self.circuit.ry(-math.pi / 2, self.qubits[i])
                    measured_qubits.append(self.qubits[i])
                    measured_bits.append(self.classical_bits[i])
                elif combined_directions[i] == 'Y':  # Do an Y measurement
                    self.circuit.rx(math.pi / 2, self.qubits[i])
                    measured_qubits.append(self.qubits[i])
                    measured_bits.append(self.classical_bits[i])
                elif combined_directions[i] == 'Z':  # Do an Z measurement
                    measured_qubits.append(self.qubits[i])
                    measured_bits.append(self.classical_bits[i])

            self.circuit.measure(measured_qubits, measured_bits)
            experimental_job = qk.execute(self.circuit, self.backend, shots=self.shots)
            experimental_result = experimental_job.result()
            histogram = experimental_result.get_counts(self.circuit)

            expected_value = 0
            for state, counts in histogram.items():
                bitsum = 0
                for bit in state:
                    bitsum += int(bit)
                expected_value += ((-1)**bitsum)*int(counts)
            expected_value = expected_value / self.shots
            pauli_output_vector[index] = expected_value
            index += 1

        return pauli_output_vector[:, 0]

    def compute_pauli_output_matrix(self):
        """
        This function simply returns an array containing all possible output Pauli State vectors. It does this by
        calling the compute_pauli_output_vector() on every input which was found with compute_all_qubit_inputs(). In
        this function there is also a bar which is used to create a progress bar when the main.py is run in a terminal.
        This is quite handy as calculation time dramatically increases with the amount of qubits n.
        :return: pauli_output_matrix
        """
        pauli_output_matrix = self.pauli_output_matrix

        state_index = 0
        bar = IncrementalBar(': Computing output matrix...', max=len(self.qubit_inputs))  # For tracking progress
        for new_input in self.qubit_inputs:
            pauli_output_matrix[:, state_index] = self.compute_pauli_output_vector(new_input)
            state_index += 1
            bar.next()  # For tracking progress
        bar.finish()  # Finished the big calculation!

        self.pauli_output_matrix = pauli_output_matrix
        return pauli_output_matrix

    def compute_theoretical_output_vector(self, qubit_states):
        """
        This function is similar to the calculation for the pauli input/output vectors. But now we use a specific
        statevector_simulator backend. This allows to get a theoretical output vector, which has no noise.
        :return: theoretical_output_vector, this is the theoretical Pauli state vector of the output.
        """
        sv_backend = Aer.get_backend('statevector_simulator')  # statevector for theoretical transfer matrix
        theoretical_output_vector = np.zeros((len(self.directions) ** self.n, 1))

        index = 0
        for combined_directions in self.pauli_vector_string:
            # Start with clearing out the old circuit, setting up and adding the process
            self.clear_circuit()
            self.setup_circuit(qubit_states)
            self.quantum_process()

            # Calculate the directional_matrix
            directional_matrix = self.get_pauli_matrix(list(combined_directions)[0])  # Pick the first element
            for char in list(combined_directions)[1:]:  # calc. the kron prod. for all possible directional matrices
                directional_matrix = np.kron(self.get_pauli_matrix(char), directional_matrix)

            # Now do a statevector analysis to determine the theoretical output
            statevector_job = qk.execute(self.circuit, sv_backend)
            statevector_result = statevector_job.result()
            statevector_output = statevector_result.get_statevector(self.circuit, decimals=3)
            theoretical_output_vector[index, 0] = np.real(np.dot(np.conjugate(statevector_output),
                                                                 np.dot(directional_matrix, statevector_output)))
            index += 1

        return theoretical_output_vector[:, 0]

    def compute_theoretical_output_matrix(self):
        """
        This function simply returns an array containing all possible theoretical output Pauli State vectors. It does
        this by calling the theoretical_output_vector() on every input which was found with compute_all_qubit_inputs().
        :return: theoretical_output_matrix
        """
        theoretical_output_matrix = self.theoretical_output_matrix

        state_index = 0
        bar = IncrementalBar(': Computing theoretical output matrix...', max=len(self.qubit_inputs))  # For progress
        for new_input in self.qubit_inputs:
            theoretical_output_matrix[:, state_index] = self.compute_theoretical_output_vector(new_input)
            state_index += 1
            bar.next()  # For tracking progress
        bar.finish()  # Finished the calculation

        self.theoretical_output_matrix = theoretical_output_matrix
        return theoretical_output_matrix

    def compute_pauli_transfer_matrix(self):
        """
        Now that we have done most of the complicated algorithms this one can be rather simple. Using the
        pauli_input_matrix and pauli_output_matrix we can calculate the Pauli Transfer Matrix which characterizes the
        Quantum Process which was given in quantum_process().
        :return: pauli_transfer_matrix
        """
        pauli_input_matrix = self.compute_pauli_input_matrix()
        inverse_pauli_input_matrix = np.linalg.pinv(pauli_input_matrix)
        pauli_output_matrix = self.compute_pauli_output_matrix()
        pauli_transfer_matrix = pauli_output_matrix * np.matrix(inverse_pauli_input_matrix)
        self.pauli_transfer_matrix = pauli_transfer_matrix
        return pauli_transfer_matrix

    def compute_theoretical_transfer_matrix(self):
        """
        Similar to compute_pauli_transfer_matrix(), calculating the theoretical_transfer_matrix is now rather simple.
        Using the pauli_input_matrix and theoretical_output_matrix we can calculate the Theoretical Transfer Matrix
        which characterizes the Quantum Process perfectly.
        :return: pauli_transfer_matrix
        """
        pauli_input_matrix = self.compute_pauli_input_matrix()
        inverse_pauli_input_matrix = np.linalg.pinv(pauli_input_matrix)
        theoretical_output_matrix = self.compute_theoretical_output_matrix()
        theoretical_transfer_matrix = theoretical_output_matrix * np.matrix(inverse_pauli_input_matrix)
        self.theoretical_transfer_matrix = theoretical_transfer_matrix
        return theoretical_transfer_matrix

    def compute_fidelity(self):
        """
        In this function the gate fidelity is calculated. Do a call to compute_pauli_transfer_matrix() and
        compute_theoretical_transfer_matrix() before this.
        :return: gate_fidelity
        """
        dimension = 2 ** self.n
        process_fidelity = (np.trace(np.dot(np.matrix(self.theoretical_transfer_matrix).transpose(),
                                            self.pauli_transfer_matrix))) / dimension ** 2
        average_gate_fidelity = (process_fidelity * dimension + 1) / (dimension + 1)

        return average_gate_fidelity

    def plot_pauli_transfer_matrix(self, plot_title):
        """
        Using this we can visualize the QPT in a colour map plot.
        :param plot_title: The title you want above the plot
        :return: none
        """
        labelling = self.pauli_vector_string

        fig, ax = plt.subplots()
        ax.imshow(self.pauli_transfer_matrix, vmin=-1, vmax=1, interpolation='nearest', cmap=cm.RdBu)
        # We want to show all ticks...
        ax.set_xticks(np.arange(len(labelling)))
        ax.set_yticks(np.arange(len(labelling)))
        # ... and label them with the respective list entries
        ax.set_xticklabels(labelling)
        ax.set_yticklabels(labelling)
        # Rotate the x-tick labels so they don't overlap
        plt.xticks(rotation=90)

        ax.set_title(plot_title)
        plt.show(block=False)

    def plot_theoretical_transfer_matrix(self, plot_title):
        """
        Using this we can visualize the theoretical QPT in a colour map plot.
        :param plot_title: The title you want above the plot
        :return: none
        """
        labelling = self.pauli_vector_string

        fig, ax = plt.subplots()
        ax.imshow(self.theoretical_transfer_matrix, vmin=-1, vmax=1, interpolation='nearest', cmap=cm.RdBu)
        # We want to show all ticks...
        ax.set_xticks(np.arange(len(labelling)))
        ax.set_yticks(np.arange(len(labelling)))
        # ... and label them with the respective list entries
        ax.set_xticklabels(labelling)
        ax.set_yticklabels(labelling)
        # Rotate the x-tick labels so they don't overlap
        plt.xticks(rotation=90)

        ax.set_title(plot_title)
        plt.show(block=False)


# -------------------------------------- End of the QubitProcessTomography class --------------------------------------
