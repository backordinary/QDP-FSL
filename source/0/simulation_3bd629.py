# https://github.com/Baccios/CTC_iterator/blob/02c77c2a8e882e48df6ef1dcd186f9124ecaad38/ctc/simulation.py
"""
This module provides an API to perform simulations of an iterated D-CTC simulation circuit.
"""

import math
import os
from math import pi, sqrt

import scipy.stats

import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister, execute, transpile
from qiskit.circuit import Gate
from qiskit.extensions import Initialize
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.ibmq import IBMQBackend
from qiskit.tools.monitor import job_monitor
from qiskit.providers.ibmq.managed import IBMQJobManager

# import basic plot tools
import matplotlib.pyplot as plt

# from ctc.block_generator import get_ctc_assisted_circuit
from ctc.brun import get_3d_code
from ctc.encodings import get_brun_fig2_encoding, get_2d_code
from ctc.gates.cloning import CloningGate
from ctc.gates.ctc_assisted import CTCGate
from ctc.gates.p_gate import PGate


class CTCCircuitSimulator:
    """
    This class provides tools to generate, run and visualize results of a
    simulation of a CTC assisted iterated circuit.
    """

    def __init__(self, size, k_value, cloning_size=7, ctc_recipe="nbp", base_block=None):
        """
        Initialize the simulator with a set of parameters.

        :param size: The number of bits used to encode k
        :type size: int
        :param k_value: The value for k, must be lower than 2^size
        :type k_value: int
        :param ctc_recipe: the recipe used to build the ctc assisted gate. It defaults to "nbp".
                   Possible values are:
                    <ol>
                        <li>"nbp": use the algorithm in
                            <a href="https://arxiv.org/abs/1901.00379">this article</a> (default value)
                        </li>
                        <li>"brun": use the algorithm in
                            <a href="https://arxiv.org/abs/0811.1209">this article</a> and apply the 2d encoding with
                            states equally distributed on the XZ plane of the Bloch sphere
                        </li>
                        <li>"brun_fig2": use the algorithm in
                            <a href="https://arxiv.org/abs/0811.1209">this article</a> in the variant for Fig.2.
                            (Note that this case is only limited to 2 bits)
                        </li>
                        <li>
                            "brun_3d": use the algorithm in
                            <a href="https://arxiv.org/abs/0811.1209">this article</a> and apply the 3d encoding
                            scheme for input states.
                        </li>
                        <li>
                            "brun_quadrant": use the algorithm in
                            <a href="https://arxiv.org/abs/0811.1209">this article</a> and apply the 2d encoding with
                            states confined in a 45 degrees quadrant of the Bloch sphere
                        </li>
                    </ol>
        :type ctc_recipe: str
        :param base_block: The explicit gate to be used as basic block. When not specified,
                the gate will be built using the recipe specified in parameter ctc_recipe.
        :type base_block: qiskit.circuit.Gate
        :param cloning_size: the size of the internal CloningGate. It will be significant
        only in case approximate quantum cloning is used in iterations.
        :type cloning_size: int
        """

        if size <= 0:
            raise ValueError("parameter size must be greater than zero")
        if k_value < 0 or k_value > (2 ** size - 1):
            raise ValueError("parameter k_value must be between zero and 2^size - 1")

        self._size = size
        self._k_value = k_value
        self._cloning_size = cloning_size
        self._ctc_recipe = ctc_recipe
        if base_block is None:
            self._ctc_gate = CTCGate(2 * size, method=ctc_recipe, label="V Gate")
        else:
            if not isinstance(base_block, Gate):
                raise TypeError("parameter base_block must be a Gate")
            self._ctc_gate = base_block

    def _build_dctr(self, iterations, cloning="no_cloning", cloning_method="eqcm_xz"):
        """
        Build the dctr circuit using instance initialization parameters (utility)
        :param iterations: The number of iterations to build
        :type iterations: int
        :param cloning: can assume one of these values:
                    <ol>
                        <li>"no_cloning" (default): Do not use cloning to replicate psi state</li>
                        <li>"initial": use cloning only for the first cloning_size iterations</li>
                        <li>"full": use cloning for each iteration
                    </ol>
        :type cloning: str
        :param cloning_method: The cloning method to use:
            <ul>
                <li>"uqcm" (default): use the Universal Quantum Cloning Machine</li>
                <li>"eqcm_xz": use cloning machine optimized for equatorial qubits on x-z plane</li>
            </ul>
        :type cloning_method: str
        :return: the full circuit ready to run.
        :rtype: qiskit.circuit.Instruction
        """
        size = self._size

        init_gate = self._get_psi_init()

        qubits = QuantumRegister(2 * size)
        clone_qubits = QuantumRegister(self._cloning_size)
        dctr_circuit = QuantumCircuit(qubits)

        if cloning == "no_cloning":
            # initialize the first of CR qubits with psi
            dctr_circuit.append(init_gate, [qubits[size]])
        else:
            dctr_circuit.add_register(clone_qubits)
            # initialize the cloning circuit
            dctr_circuit.append(init_gate, [clone_qubits[0]])
            dctr_circuit.append(
                CloningGate(self._cloning_size, method=cloning_method),
                clone_qubits
            )
            # use the first clone to initialize psi
            dctr_circuit.swap(qubits[size], clone_qubits[1])

        # useful to update next_clone_index
        def update_index(i):
            res = (i + 1) % self._cloning_size
            # print("Next clone index is: " + str(res))  # DEBUG
            return res

        next_clone_index = update_index(1)  # used to keep track of the clone to be used

        # place the initial CTC gate
        dctr_circuit.append(self._ctc_gate, qubits)

        # behavior depends on the value of cloning attribute:
        if cloning == "no_cloning":
            # in this case we simply apply default iterations with the real state
            iteration_instruction = self._get_iteration()
            for _ in range(iterations - 1):
                dctr_circuit.append(iteration_instruction, qubits)

        elif cloning == "initial":
            # in this case we use clones as long as they are available (self._cloning_size - 1)
            iteration_instruction = self._get_iteration()
            for _ in range(iterations - 1):
                if next_clone_index == 0:  # if clones are finished we use the real state
                    dctr_circuit.append(iteration_instruction, qubits)
                else:
                    dctr_circuit.append(
                        self._get_iteration(psi_qubit=clone_qubits[next_clone_index]),
                        qubits[:] + clone_qubits[:]
                    )
                    next_clone_index = update_index(next_clone_index)

        elif cloning == "full":
            # in this case we always use clones. If they are finished, we reset and clone again
            for _ in range(iterations - 1):
                if next_clone_index == 0:  # in this case we need to re init clones
                    for qubit in clone_qubits:
                        dctr_circuit.reset(qubit)
                    dctr_circuit.append(init_gate, [clone_qubits[0]])
                    dctr_circuit.append(
                        CloningGate(self._cloning_size, method=cloning_method),
                        clone_qubits
                    )
                    dctr_circuit.append(
                        self._get_iteration(psi_qubit=clone_qubits[1]), qubits[:] + clone_qubits[:]
                    )
                    next_clone_index = 2
                else:
                    dctr_circuit.append(
                        self._get_iteration(psi_qubit=clone_qubits[next_clone_index]),
                        qubits[:] + clone_qubits[:]
                    )
                    next_clone_index = update_index(next_clone_index)

        else:
            raise ValueError("cloning must be either \"no_cloning\", \"initial\" or \"full\"")

        # dctr_circuit.draw(output="mpl")  # DEBUG
        # plt.show()

        # print("REQUIRED " + str(len(dctr_circuit.qubits)))  # DEBUG

        return dctr_circuit.to_instruction()

    def _get_psi_init(self):
        """
        Get an Initialize gate for psi state
        :return: the gate used to initialize psi
        :rtype: qiskit.circuit.Gate
        """

        # first treat the alternative encoding for Brun (Fig 2 case):
        if self._ctc_recipe == "brun_fig2":
            psi = get_brun_fig2_encoding(self._k_value, self._size).tolist()

        elif self._ctc_recipe == "brun_3d":
            psi = get_3d_code(self._k_value, self._size).tolist()

        elif self._ctc_recipe == "brun_quadrant":
            psi = get_2d_code(self._k_value, self._size, sector_divider=2**(self._size + 2)).tolist()

        else:
            # encode k in a state |ψ⟩ = cos(kπ/2^n)|0⟩ + sin(kπ/2^n)|1⟩
            psi = get_2d_code(self._k_value, self._size).tolist()

        # print("k = ", self.__k_value, ", psi is encoded as: ", psi)  # DEBUG

        # Let's create our initialization gate to create |ψ⟩
        init_gate = Initialize(psi)
        init_gate.label = "psi-init"
        return init_gate

    def _get_iteration(self, psi_qubit=None):
        """
        Get a single iteration of the circuit as a gate

        :param psi_qubit: The qubit containing psi to use as input to the CTC gate.
                             If set to None, psi is initialized using Initialize()
        :type psi_qubit: qiskit.circuit.Qubit
        :return: The iteration gate
        :rtype: qiskit.circuit.Instruction
        """
        size = self._size

        qubits = QuantumRegister(2 * size)

        iter_circuit = QuantumCircuit(qubits)

        if psi_qubit is not None:
            iter_circuit.add_register(psi_qubit.register)

        init_gate = self._get_psi_init()

        for i in range(size):
            iter_circuit.swap(qubits[i], qubits[size + i])

        iter_circuit.barrier()
        
        # reset CR qubits
        for i in range(size, 2 * size):
            iter_circuit.reset(i)

        # initialize the first CTC assisted gate
        if psi_qubit is None:
            iter_circuit.append(init_gate, [qubits[size]])
        else:
            iter_circuit.swap(qubits[size], psi_qubit)
        iter_circuit.append(self._ctc_gate, qubits)

        # iter_circuit.draw(output="mpl")  # DEBUG
        # plt.show()

        return iter_circuit.to_instruction()

    @staticmethod
    def _check_binary(c_value_string):
        """
        Check if a string contains only binary characters (utility)

        :param c_value_string: The string to check
        :return: True if the string is binary
        """
        set_value = set(c_value_string)
        if set_value in ({'0', '1'}, {'0'}, {'1'}):
            return True
        return False

    # takes in input c either as a binary string (little endian) (e.g. '1001')
    # or as an integer and initializes the circuit with such CBS state
    def _initialize_c_state(self, c_value, dctr_circuit):
        """
        Initialize ancillary qubits using a binary string or an integer.
        takes in input c either as a binary string (little endian) (e.g. '1001')
        or as an integer and initializes the circuit with such CBS state. (utility)

        :param c_value: the value to initialize
        :type c_value: str, int
        :param dctr_circuit: The quantum circuit to be initialized
        :type dctr_circuit: qiskit.circuit.quantumcircuit.QuantumCircuit
        :return: None
        """

        if isinstance(c_value, int):
            c_value = format(c_value, '0' + str(self._size) + 'b')
        if isinstance(c_value, str):
            string_state = c_value
            if (len(string_state) != self._size) or not self._check_binary(string_state):
                raise ValueError("c string must be binary, e.g. \"10001\"")

            for i, bit in enumerate(string_state):
                if bit == '1':
                    dctr_circuit.x(i)
        elif isinstance(c_value, list):
            # in this case c_value must be a list of state vectors
            for i, state in enumerate(c_value):
                if not isinstance(state, list):
                    raise ValueError("c value was a list not containing state vectors lists")
                dctr_circuit.append(Initialize(state), [dctr_circuit.qubits[i]])

        elif isinstance(c_value, float):
            p_gate = PGate(c_value)
            # in this case c_value must be a probability
            for i in range(self._size):
                dctr_circuit.append(p_gate, [dctr_circuit.qubits[i]])
        else:
            raise TypeError("c value was not a string, an integer or a list")

    @staticmethod
    def _get_params(**params):
        """
        Utility method to initialize kwargs
        """
        cloning = params.get("cloning", "no_cloning")
        backend = params.get("backend", QasmSimulator())
        cloning_method = params.get("cloning_method", "eqcm_xz")
        shots = params.get("shots", 1024)

        return cloning, backend, cloning_method, shots

    def simulate(self, c_value, iterations, **params):
        """
        Simulate a run of the CTC iterated circuit

        :param c_value: The value for the ancillary qubits, either as binary string or integer, list or float.
                        If integer, must be between 0 and 2^size - 1.
                        If string, must be a binary string with size digits
                        If float, it must be the probability of reading zero, and c state will
                        be sqrt(c_value)|0⟩ + sqrt(1 - c_value)|1⟩
                        Example: <code>c_value="0111"</code>
                        is the same as <code>c_value=5</code> and
                        <code>c_value=[[1,0],[0,1],[0,1],[0,1]]</code>
        :type c_value: str, int, list, float
        :param iterations: the number of iterations to simulate
        :type iterations: int
        :param params:
            List of accepted parameters:
            <ul>
              <li>
                cloning: can assume one of these values:
                <ol>
                    <li>"no_cloning" (default): Do not use cloning to replicate psi state</li>
                    <li>"initial": use cloning only for the first cloning_size iterations</li>
                    <li>"full": use cloning for each iteration
                </ol>
              </li>
              <li>
                backend: The backend to use, defaults to QasmSimulator()
              </li>
              <li>
                shots: The number of shots for the simulation. Defaults to 512
              </li>
              <li>
                cloning_method: the cloning method. Can be one of:
                <ol>
                    <li>"uqcm": use the Universal Quantum Cloning Machine</li>
                    <li>
                        "eqcm_xz" (default):
                        use cloning machine optimized for equatorial qubits on x-z plane</li>
                </ol>
              </li>
            </ul>
        :return: The count list of results
        :rtype: dict
        """

        cloning, backend, cloning_method, shots = self._get_params(**params)

        dctr_simulation_circuit = \
            self._build_simulator_circuit(
                c_value, iterations, cloning=cloning, cloning_method=cloning_method
            )

        # print("I'm about to execute ", iterations, " iterations...")  # DEBUG
        job = execute(dctr_simulation_circuit, backend, shots=shots, optimization_level=0)
        # circ = transpile(dctr_simulation_circuit, backend=backend, optimization_level=0)
        # my_qobj = assemble(circ)
        # job = backend.run(my_qobj, shots=shots)

        # job_monitor(job) # DEBUG

        counts = job.result().get_counts()

        # plot_histogram(counts)  # DEBUG
        # plot.show()

        return counts

    def _build_simulator_circuit(self, c_value, iterations,
                                 cloning="no_cloning",
                                 cloning_method="eqcm_xz",
                                 add_measurements=True):
        """
        Build a dctr QuantumCircuit ready for a simulation (utility)
        :param c_value: the initial value for ancillary qubits
        :type c_value: int, str, list, float
        :param iterations: the number of iterations in the circuit
        :type iterations: int
        :param cloning: can assume one of these values:
                    <ol>
                        <li>"no_cloning" (default): Do not use cloning to replicate psi state</li>
                        <li>"initial": use cloning only for the first cloning_size iterations</li>
                        <li>"full": use cloning for each iteration
                    </ol>
        :type cloning: str
        :param cloning_method: The cloning method to use:
            <ul>
                <li>"uqcm" (default): use the Universal Quantum Cloning Machine</li>
                <li>"eqcm_xz": use cloning machine optimized for equatorial qubits on x-z plane</li>
            </ul>
        :type cloning_method: str
        :param add_measurements: if True, also adds measurements at the end of the circuit
        :type add_measurements: bool
        :return: The circuit ready to be executed
        :rtype: qiskit.circuit.QuantumCircuit
        """

        if iterations <= 0:
            raise ValueError("parameter iterations must be greater than zero")

        dctr_instr = self._build_dctr(iterations, cloning, cloning_method=cloning_method)

        # initialize the final circuit
        classical_bits = ClassicalRegister(self._size)

        num_qubits = dctr_instr.num_qubits
        if cloning != "no_cloning":
            num_qubits -= self._cloning_size

        qubits = QuantumRegister(num_qubits)

        dctr_simulation_circuit = QuantumCircuit(qubits, classical_bits)

        if cloning != "no_cloning":
            clone_qubits = QuantumRegister(self._cloning_size)
            dctr_simulation_circuit.add_register(clone_qubits)
            # initialize ancillary qubits
            self._initialize_c_state(c_value, dctr_simulation_circuit)

            # print("GIVEN " + str(len(qubits[:] + clone_qubits[:])))  # DEBUG

            dctr_simulation_circuit.append(dctr_instr, qubits[:] + clone_qubits[:])

        else:
            # initialize ancillary qubits
            self._initialize_c_state(c_value, dctr_simulation_circuit)
            dctr_simulation_circuit.append(dctr_instr, qubits)

        if add_measurements:
            # noinspection PyTypeChecker
            self._add_measurement(dctr_simulation_circuit)

        return dctr_simulation_circuit

    def _add_measurement(self, dctr_circuit):
        classical_bits = dctr_circuit.clbits
        qubits = dctr_circuit.qubits
        for i in range(self._size):
            dctr_circuit.measure(qubits[i + self._size],
                                 classical_bits[self._size - 1 - i])

    def _binary(self, value):
        """
        Formats value as a binary string on size digits (utility)

        :param value: the integer to be formatted
        :return: the binary string
        """
        return ('{0:0' + str(self._size) + 'b}').format(value)

    def test_convergence(self, c_value, start, stop, step=2, **params):
        """
        Test the convergence rate of the algorithm by simulating it
        under an increasing number of iterations. Save the output plot in ./out

        :param c_value: The value for the ancillary qubits, either as binary string or integer, list or float.
                        If integer, must be between 0 and 2^size - 1.
                        If string, must be a binary string with size digits
                        If float, it must be the probability of reading zero, and c state will
                        be sqrt(c_value)|0⟩ + sqrt(1 - c_value)|1⟩
                        Example: <code>c_value="0111"</code>
                        is the same as <code>c_value=5</code> and
                        <code>c_value=[[1,0],[0,1],[0,1],[0,1]]</code>
        :type c_value: str, int, list, float
        :param start: the starting number of iterations
        :type start: int
        :param stop: the final number of iterations
        :type stop: int
        :param step: the iterations increase step
        :type step: int
        :param params:
            List of accepted parameters:
            <ul>
              <li>
                cloning: can assume one of these values:
                <ol>
                    <li>"no_cloning" (default): use only fresh copies to replicate psi state</li>
                    <li>"initial": use quantum cloning only for the first cloning_size iterations</li>
                    <li>"full": use cloning for each iteration
                </ol>
              </li>
              <li>
                cloning_method: the cloning method. Can be one of:
                <ol>
                    <li>"uqcm": use the Universal Quantum Cloning Machine</li>
                    <li>
                        "eqcm_xz" (default):
                        use cloning machine optimized for equatorial qubits on x-z plane
                    </li>
                </ol>
              </li>
              <li>
                backend: The backend to use, defaults to QasmSimulator()
              </li>
              <li>
                shots: The number of shots for the simulation. Defaults to 1024
              </li>
            </ul>
        :return: None
        """
        cloning, backend, cloning_method, shots = self._get_params(**params)

        iterations = list(range(start, stop + 1, step))

        probabilities, conf_intervals_95 = self._execute_convergence(
            c_value,
            iterations,
            cloning=cloning,
            backend=backend,
            cloning_method=cloning_method,
            shots=shots
        )

        self._plot_convergence(c_value, probabilities, conf_intervals_95, iterations)

    def _execute_convergence(self, c_value, iterations, **params):

        cloning = params.get("cloning", "no_cloning")
        backend = params.get("backend", QasmSimulator())
        cloning_method = params.get("cloning_method", "eqcm_xz")
        shots = params.get("shots", 1024)

        # if the backend is an IBMQ,
        # we submit all circuits together to optimize waiting time using IBMQJobManager
        if isinstance(backend, IBMQBackend):

            print("Building the circuits to submit...")
            circuits = [
                self._build_simulator_circuit(c_value, i, cloning=cloning,
                                              cloning_method=cloning_method)
                for i in iterations
            ]

            # Need to transpile the circuits first for optimization
            circuits = transpile(circuits, backend=backend)
            print("Circuits built and transpiled!")
            # circuits[len(circuits) - 1].draw(output="mpl", filename="./test.png")  # DEBUG

            # Use Job Manager to break the circuits into multiple jobs.
            job_manager = IBMQJobManager()
            job_set = job_manager.run(circuits, backend=backend, name='test_convergence')

            # we monitor the first job in the list to have some feedback
            job_monitor(job_set.jobs()[0])

            results = job_set.results()

            # extract normalized probabilities of success
            probabilities = [
                results.get_counts(i)[self._binary(self._k_value)] / shots
                for i in range(len(circuits))
            ]

            conf_intervals_95 = [
                scipy.stats.norm.ppf(0.975) * sqrt(probabilities[i]
                                                   * (1 - probabilities[i]) / shots)
                for i in range(len(circuits))
            ]

        else:
            probabilities = []
            conf_intervals_95 = []

            for i in iterations:
                count = self.simulate(
                    c_value=c_value,
                    iterations=i,
                    cloning=cloning,
                    cloning_method=cloning_method,
                    backend=backend,
                    shots=shots
                )

                stop = iterations[len(iterations) - 1]

                print(
                    "simulation ended: " + str(i) + " iterations (" + str(i) + "/" + str(stop) + ")"
                )

                norm_shots = sum(count.values())  # should be equal to shots

                correct_key = self._binary(self._k_value)

                if correct_key in count.keys():
                    success_prob = count[correct_key] / norm_shots
                else:
                    success_prob = 0

                confidence_int_95 = scipy.stats.norm.ppf(0.975) * \
                    sqrt(success_prob * (1 - success_prob) / norm_shots)

                probabilities.append(success_prob)
                conf_intervals_95.append(confidence_int_95)

                print(
                    "   -> probability = ",  success_prob,  " +- ",  confidence_int_95
                )
                print()

        return probabilities, conf_intervals_95

    def _plot_convergence(self, c_value, probabilities, conf_intervals, iterations, output="out"):
        # select the first plot
        plt.figure(1)

        plt.ylabel('Probability')
        plt.xlabel('Iterations')

        title = "Bar plot: n_bits = " + str(self._size)
        filename = str(self._size) + "_"

        if isinstance(c_value, int):
            title += ", initial_state |" + self._binary(c_value) + "⟩"
            filename += "initial_" + self._binary(c_value)
        elif isinstance(c_value, str):
            title += ", initial_state |" + c_value + "⟩"
            filename += "initial_" + c_value

        if isinstance(self._k_value, int):
            title += ", target |" + self._binary(self._k_value) + "⟩"
            filename += "_target_" + self._binary(self._k_value)
        elif isinstance(self._k_value, str):
            title += ", target |" + self._k_value + "⟩"
            filename += "_target_" + self._k_value

        plt.title(title)

        plt.axhline(y=0.95, linewidth=1, color='red')

        # build the bar plot
        x_positions = np.arange(len(probabilities))
        plt.bar(
            x_positions,
            probabilities,
            color='blue',
            edgecolor='black',
            yerr=conf_intervals,
            capsize=7,
            label='success probability'
        )

        plt.xticks(x_positions, iterations)
        plt.grid(axis='y', linestyle='--', linewidth=0.5)

        if not os.path.exists(output):
            try:
                os.makedirs(output)
            except OSError as _:
                print("Error: could not create \"" + output + "\" directory")
                return

        # finally save the plot
        image_basename = output + '/' + filename
        plt.savefig(image_basename + '_bar.pdf')
        plt.close()

        """
        # select the second plot
        plt.figure(2)

        plt.ylabel('Probability')
        plt.xlabel('Iterations')

        if isinstance(c_value, int):
            plt.title("Convergence log-log rate: n_qbits = " + str(self._size) +
                      ", initial_state |" + self._binary(c_value) + "⟩")
        elif isinstance(c_value, str):
            plt.title("Convergence log-log rate: n_qbits = " + str(self._size) +
                      ", initial_state |" + c_value + "⟩")
        else:
            plt.title("Convergence log-log rate: n_qbits = " + str(self._size))

        # build log-log plot
        plt.errorbar(
            iterations, probabilities, fmt='o', yerr=conf_intervals, label='success_probability'
        )
        plt.loglog(iterations, probabilities, color='blue')
        plt.xscale('log', base=2)
        plt.yscale('log', base=2)
        plt.grid()

        # save also the second plot
        plt.savefig(image_basename + '_log.pdf')
        plt.close()
        """

    def test_c_impact(self, c_values, start, stop, step=2, c_tick_labels=None, **params):
        """
        Test the convergence of the algorithm with different initial values of c.
        Plot the outcome in a 2d or 3d bar plot.
        Save the output plot in ./out

        :param c_value: The value for the ancillary qubits, either as binary string or integer, list or float.
                        If integer, must be between 0 and 2^size - 1.
                        If string, must be a binary string with size digits
                        If float, it must be the probability of reading zero, and c state will
                        be sqrt(c_value)|0⟩ + sqrt(1 - c_value)|1⟩
                        Example: <code>c_value="0111"</code>
                        is the same as <code>c_value=5</code> and
                        <code>c_value=[[1,0],[0,1],[0,1],[0,1]]</code>
        :type c_value: str, int, list, float
        :param start: the starting number of iterations
        :type start: int
        :param stop: the final number of iterations
        :type stop: int
        :param step: the iterations increase step
        :type step: int
        :param c_tick_labels: labels to set to c values in the output plot
        :type c_tick_labels: list
        :param params:
            List of accepted parameters:
            <ul>
              <li>
                cloning: can assume one of these values:
                <ol>
                    <li>"no_cloning" (default): Do not use cloning to replicate psi state</li>
                    <li>"initial": use cloning only for the first cloning_size iterations</li>
                    <li>"full": use cloning for each iteration
                </ol>
              </li>
              <li>
                cloning_method: the cloning method. Can be one of:
                <ol>
                    <li>"uqcm": use the Universal Quantum Cloning Machine</li>
                    <li>
                        "eqcm_xz" (default):
                        use cloning machine optimized for equatorial qubits on x-z plane
                    </li>
                </ol>
              </li>
              <li>
                backend: The backend to use, defaults to QasmSimulator()
              </li>
              <li>
                shots: The number of shots for the simulation. Defaults to 1024
              </li>
              <li>
                plot_d: the number of dimensions of the plot. Can be 2 or 3
              </li>
            </ul>
        :return: None
        """
        cloning, backend, cloning_method, shots = self._get_params(**params)
        plot_d = params.get("plot_d", 2)

        iterations = list(range(start, stop + 1, step))

        prob_matrix = []
        conf_matrix = []

        for c_value in c_values:
            probabilities, conf_intervals_95 = self._execute_convergence(
                c_value,
                iterations,
                cloning=cloning,
                backend=backend,
                cloning_method=cloning_method,
                shots=shots
            )
            prob_matrix.append(probabilities)
            conf_matrix.append(conf_intervals_95)

        if plot_d == 3:
            self._plot_c_variability_3d(prob_matrix, iterations, c_tick_labels)
        else:
            self._plot_c_variability_2d(prob_matrix, iterations, c_tick_labels)

    def _plot_c_variability_3d(self, prob_matrix, iterations, c_tick_labels=None, output="out"):

        y_names = iterations

        if c_tick_labels is not None:  # if c names are set use them
            x_names = c_tick_labels
        else:  # otherwise use incremental integers
            x_names = np.arange(0, len(prob_matrix), 1)

        x_pos = np.arange(0, len(x_names), 1)
        y_pos = np.arange(0, len(y_names), 1)
        x_pos, y_pos = np.meshgrid(x_pos + 0.25, y_pos + 0.25)  # leave some space between bars
        x_pos = x_pos.flatten()
        y_pos = y_pos.flatten()
        z_pos = np.zeros(len(x_names) * len(y_names))

        bar_width = 0.5 * np.ones_like(z_pos)
        bar_depth = 0.5 * np.ones_like(z_pos)
        bar_height = np.array(prob_matrix).transpose().flatten()

        fig = plt.figure()
        ax = Axes3D(fig)

        ax.bar3d(x_pos, y_pos, z_pos, bar_width, bar_depth, bar_height)

        ax.w_xaxis.set_ticks(np.arange(0.5, len(x_names), 1))
        ax.w_xaxis.set_ticklabels(x_names)
        ax.w_yaxis.set_ticks(np.arange(0.5, len(y_names), 1))
        ax.w_yaxis.set_ticklabels(y_names)
        ax.set_ylabel('Iterations')
        ax.set_xlabel('c values')
        ax.set_zlabel('Probability')
        ax.set_title("Convergence Bars: num_qbits= "
                     + str(self._size) + ", |k⟩ = |" + self._binary(self._k_value) + "⟩", y=1.0)

        # save the plot
        if not os.path.exists(output):
            try:
                os.makedirs(output)
            except OSError as _:
                print("Error: could not create \"" + output + "\" directory")
                return
        output_file = output + "/c_3d_bar_" + str(self._size) +\
            "_qbits_k=" + self._binary(self._k_value) + ".pdf"
        plt.savefig(output_file)
        plt.close()

    def _plot_c_variability_2d(self, prob_matrix, iterations, c_tick_labels=None, output="out"):

        fig = plt.figure()
        ax = fig.add_subplot(111)

        colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']

        width = (2/len(prob_matrix))*(iterations[len(iterations) - 1])/17

        x_pos = np.array([iterations[i] + width*i for i in range(0, len(iterations))])

        rects = [ax.bar(x_pos + width*i, series, width=width, color=colors[i % len(colors)])[0]
                 for i, series in enumerate(prob_matrix)]

        if c_tick_labels is not None:
            ax.legend(rects, c_tick_labels)

        ax.set_ylabel("Success probability")
        ax.set_xlabel("Iterations")
        ax.set_xticks(x_pos + (width*len(prob_matrix)/2) - 0.5*width)
        ax.set_xticklabels(iterations)
        ax.set_title("Convergence Bars: num_bits= "
                     + str(self._size) + ", |k⟩ = |" + self._binary(self._k_value) + "⟩")

        output_file = output + "/c_2d_bar_" + str(self._size) + \
            "_bits_k=" + self._binary(self._k_value) + ".pdf"
        plt.savefig(output_file)
        plt.close()
