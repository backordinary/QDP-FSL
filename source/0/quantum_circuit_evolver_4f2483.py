# https://github.com/shailendrabhandari/QUEVO/blob/71a484f1c71840142d0d11a30d8433a426f8466d/Archive/quantum_circuit_evolver.py
#    Copyright ""Bhandari, S., Overskott, S., Adamopoulos, I., Lind, P.G., Denysov,
#    S., Nichele, S. (2022). Evolving Quantum Circuits to Implement Stochastic and 
#    Deterministic Cellular Automata Rules. In: Chopard, B., Bandini, S., Dennunzio, A.,
#    Arabi Haddad, M. (eds) Cellular Automata. ACRI 2022. Lecture Notes in Computer 
#    Science, vol 13402. Springer, Cham. https://doi.org/10.1007/978-3-031-14926-9_11""

import copy
import math
import random
from typing import List
from qiskit import *
from scipy.special import rel_entr


class Chromosome(object):
    """
    A class used to represent a quantum computer _circuit as a list of integers.

    The _circuit is represented as a list
    of integers, where each gate is three successive integers i.e.:

    Three gates as list:
    [2, 0, 1, 3, 1 ,1, 4, 0, 2]

    Gate number:               1  |  2  |  3
    Gate int representation:  201 | 311 | 402

    The first int is what kind of gate it is (i.e. X-gate, C-NOT, Hadamard).
    The second int is what qubit is this assigned to (0 -> qubit 0, 1 -> qubit 1, ...)
    The third int is what qubit is controlling the gate. in cases where gates do not have an
    external controller, this int is ignored.

    The gates are given in the _gate_list attribute, and is hardcoded for the moment.
    The table under shows what gates are acceptable and how to notate the in the _gae

    | Supported gate types| Notation |
    |---------------------|----------|
    |       Hadamard      |    'h'   |
    |       Pauli X       |    'x'   |
    |       Pauli Z       |    'z'   |
    |       Pauli Y       |    'y'   |
    |        C-NOT        |   'cx'   |
    |      Swap gate      |  'swap'  |
    |       Toffoli       | 'toffoli'|
    |         RXX         |   'rxx'  |
    |         RZZ         |   'rzz'  |

    Some gates (RZZ, RXX) also need an angle value (theta) stored in a separate list.

    Attributes
    ----------
    _integer_list: List[int]
        List of integers representing the quantum _circuit.
    _theta_list: List[float]
        A list of angle values for the gates. This list is the same length as number of gates (len(_integer_list) / 3).
    _length: int
        The number of integers in the integer representation
    _gate_types: List[str]
        Experimental. Use to adjust how many types of quantum gates to include in the _circuit during generation.
    _gate_dict: dict
    """

    def __init__(self, gate_types: List[str]) -> None:
        """The Chromosome constructor"""
        self._integer_list: List[int] = []
        self._theta_list: List[float] = []
        self._length: int = 0
        self._gate_types = gate_types
        self._gate_dict: dict = self._create_gate_dict()

    def __repr__(self) -> str:
        """Returns desired for printing == print(_integer_list)"""
        return str(self._integer_list)

    def __len__(self) -> int:
        """Returns the number of int in _integer_list"""
        return self._length

    def __iter__(self) -> List[int]:
        """Returns the iterable _integer_list"""
        yield from self._integer_list

    def _create_gate_dict(self) -> dict:
        """Creates and return a dict of the _gate_types"""
        gate_dict: dict = {}
        for j in range(0, len(self._gate_types)):
            gate_dict[str(j)] = self._gate_types[j]
        return gate_dict

    def set_integer_list(self, integer_list: List[int]):
        """
        Changes the chromosome's integer list to the one given as parameter.

        Parameters:
           integer_list (List[int]): Quantum _circuit integer representation as list.
        """
        old_integer_list = copy.copy(self._integer_list)
        self.clear()
        for integer in integer_list:
            self._integer_list.append(integer)
        self._update_length()
        if not old_integer_list:
            self._generate_theta_list()
        else:
            self._update_theta_list(old_integer_list, self._integer_list)

    def get_gate_dict(self):
        """Returns the chromosome's _gate_dict attribute"""
        return self._gate_dict

    def get_integer_list(self) -> List[int]:
        """Returns the list of integers representing the _circuit"""
        return self._integer_list

    def _update_length(self) -> None:
        """Sets _length to the number of integers in _integer_list"""
        self._length = len(self._integer_list)

    def get_length(self) -> int:
        """Returns the number of integers in _integer_list."""
        return self._length

    def get_theta_list(self) -> List[float]:
        """Returns the list of angles in the _circuit"""
        return self._theta_list

    def _generate_theta_list(self) -> None:
        """Generates a list of angles based on the current list of integers"""
        self._theta_list.clear()
        gates = int(self._length / 3)

        for i in range(0, gates):
            int_index = i * 3
            gate = self._gate_dict[str(self._integer_list[int_index])]
            if gate in ['rzz', 'rxx']:
                theta = random.uniform(0, 2 * math.pi)
                self._theta_list.append(theta)
            else:
                self._theta_list.append(0)

    def _update_theta_list(self, old_list: List[int], new_list: List[int]) -> None:
        """
        Updates the list of theta values. Used when a _integer_list is changed.
        Takes the _integer.list before the change as old_list, and the _integer_list
        after the change as new_list.

        Parameters
        ----------
        old_list : List[int]
            The list before change to list happened.
        new_list : List[int]
            The new_list list with one or more changed integers.
        """
        gates = int(self._length / 3)
        change_list = self._change_in_theta(old_list, new_list)

        for i in range(0, gates):
            int_index = i * 3
            gate = self._gate_dict[str(self._integer_list[int_index])]
            if change_list[i] == 1 and gate in ['rzz', 'rxx']:
                theta = random.uniform(0, 2 * math.pi)
                self._theta_list[i] = theta
            elif gate in ['rzz', 'rxx']:
                continue
            else:
                self._theta_list[i] = 0

    def _change_in_theta(self, old_list, new_list) -> List[int]:
        """
        Compares the old_list to the new_list and returns a binary string where
        1 indicates a change in the theta value for that gate, 0 indicates no
        change in theta.

        Parameters
        ----------
        old_list : List[int]
            The list before change to list happened.
        new_list : List[int]
            The new_list list with one or more changed integers.

        Returns
        -------
        List[int]
            a list of 1 or 0.
        """

        binary_list = []
        for i in range(0, self._length):
            if old_list[i] == new_list[i]:
                binary_list.append(0)
            else:
                binary_list.append(1)
        return binary_list

    def clear(self) -> None:
        """Clears chromosome's lists and resets _length"""
        self._integer_list.clear()
        self._theta_list.clear()
        self._length = 0

    def generate_random_chromosome(self, gates: int) -> None:
        """
        Generates a random list of integers representing a quantum _circuit with
        the parameter "gates" number of gates

        Parameters
        ----------
        gates : int
            The number of gates in the generated _circuit representation.
        """

        self.clear()
        for i in range(gates * 3):
            if i % 3 == 0:
                self._integer_list.append(random.randrange(0, len(self._gate_types)))
            else:
                self._integer_list.append(random.randrange(0, 3))

        self._update_length()
        self._fix_duplicate_qubit_assignment()
        self._generate_theta_list()

    def mutate_chromosome(self, probability: int) -> None:
        """
        Mutates the chromosome. Mutation can be of either replacing a random gate in the chromosome
        with a randomly generated new one, or replacing the chromosome by a randomly generated new one.
        Type of mutation is selected by probability.

        Parameters
        ----------
        probability : (int) optional
            Value Between 0 and 100. The probability of replacing a gate.
            The probability of replacing the whole chromosome is 1-probability
        """

        old_integer_list = copy.copy(self._integer_list)

        if random.randrange(0, 100) <= probability:
            self._replace_gate_with_random_gate()
        else:
            self._replace_with_random_chromosome()

        self._fix_duplicate_qubit_assignment()
        self._update_theta_list(old_integer_list, self._integer_list)

    def _replace_gate_with_random_gate(self) -> None:
        """
        Randomly selects a gate in the chromosome, and replaces it with a randomly generated new one.
        """
        random_index = random.randrange(0, int(self._length/3)) * 3

        self._integer_list[random_index] = random.randrange(0, len(self._gate_types))
        self._integer_list[random_index + 1] = random.randrange(0, 3)
        self._integer_list[random_index + 2] = random.randrange(0, 3)

    def _replace_with_random_chromosome(self) -> None:
        """Clears the chromosome and randomly generates a new _integer_list"""
        gates = int(self._length/3)
        self.clear()
        self.generate_random_chromosome(gates)

    def _change_qubit_connections(self) -> None:
        """Finds randomly a gate that connects two qubits and randomly changes it connections."""
        # TODO: implement
        pass

    def _fix_duplicate_qubit_assignment(self) -> None:
        """
        Checks the chromosome for gates that connects multiple qubits.
        If the gate has an invalid connection (it is connected to itself through the randomly generated integers),
        it generates a valid configuration randomly.
        """
        gates = int(self._length / 3)

        for i in range(0, gates):
            int_index = i * 3

            if ((self._gate_dict[str(self._integer_list[int_index])] in ['cx', 'swap', 'rzz', 'rxx']) and
                    self._integer_list[int_index + 1] == self._integer_list[int_index + 2]):

                if self._integer_list[int_index + 1] == 0:
                    self._integer_list[int_index + 1] = random.randrange(1, 3)

                elif self._integer_list[int_index + 1] == 1:
                    self._integer_list[int_index + 2] = 0

                elif self._integer_list[int_index + 1] == 2:
                    self._integer_list[int_index + 2] = random.randrange(0, 2)


class Generation(object):
    """
    This class represents a collection of chromosomes. The Collection is called a
    generation.

    Attributes
    ----------
    chromosome_list: List[int]
        List of chromosomes that habits the generation.
    fitness_list: List[float]
        list of fitness scores corresponding to the chromosomes in _chromosome_list.
    _chromosomes: int
        The number of chromosomes in the generation
    _gates: int
        Number of gates in each chromosome.
    """

    def __init__(self, chromosomes: int, gates: int) -> None:
        """
        The Generation constructor.

        Parameters
        ----------
        chromosomes (int):
            Number of chromosomes in the generation.
        gates (int):
            Number of gates in each chromosome.
        """
        self.chromosome_list: List[Chromosome] = []
        self.fitness_list: List[float] = []
        self._chromosomes: int = chromosomes
        self._gates: int = gates

    def create_initial_generation(self, gate_types: List[str]) -> None:
        """
        Populates the generation with chromosomes.
        """
        self.chromosome_list.clear()
        for i in range(self._chromosomes):
            chromosome = Chromosome(gate_types)
            chromosome.generate_random_chromosome(self._gates)
            self.chromosome_list.append(chromosome)

    def create_mutated_generation(self, parent: Chromosome, probability=70) -> None:
        """
        Populates the generation with mutated chromosomes. The parent in included as the first member of the next
        generation. The mutated chromosomes uses parameter parent as source for mutation.

        Parameters
        ----------
        parent (Chromosome):
            The chromosome all mutations will be generated from.
        """

        self.chromosome_list.clear()
        self.chromosome_list.append(parent)
        for i in range(self._chromosomes-1):
            mutated_chromosome = copy.deepcopy(parent)
            mutated_chromosome.mutate_chromosome(probability)
            self.chromosome_list.append(mutated_chromosome)

    def run_generation_diff(self, desired_outcome: List[float]) -> None:
        """
        Runs the simulator for all the chromosomes in the generation and
        stores the fitness for each chromosome in fitness_list.

        Parameters
        ----------
        desired_outcome (List[float]):
            A list of the eight CA outcomes we wish to test the chromosomes against.
        """

        for chromosome in self.chromosome_list:
            circuit = Circuit(chromosome)
            chromosome_fitness = abs(circuit.find_chromosome_fitness(desired_outcome))
            self.fitness_list.append(chromosome_fitness)

    def run_generation_KL(self, desired_outcome: List[float]) -> None:
        """
        Runs the simulator for all the chromosomes in the generation and
        stores the fitness for each chromosome in fitness_list.

        Parameters
        ----------
        desired_outcome (List[float]):
            A list of the eight CA outcomes we wish to test the chromosomes against.
        """

        for chromosome in self.chromosome_list:
            circuit = Circuit(chromosome)
            chromosome_fitness = abs(circuit.find_kullback_liebler_fitness(desired_outcome))
            self.fitness_list.append(chromosome_fitness)

    def get_best_fitness(self):
        """Returns the fitness value for the best chromosome in the generation."""
        best_fitness = min(self.fitness_list)
        return best_fitness

    def get_best_chromosome(self):
        """Returns the chromosome with the best fitness in the generation."""
        best_fitness_index = self.fitness_list.index(self.get_best_fitness())
        best_chromosome = self.chromosome_list[best_fitness_index]
        return best_chromosome

    def print_chromosomes(self):
        """Prints all the generation's chromosomes."""
        print("Chromosomes: ")
        for chromosome in self.chromosome_list:
            print(chromosome)
        print('\n')

    def print_theta_values(self):
        """Prints all the generation's theta values."""
        print("Theta values: ")
        for chromosome in self.chromosome_list:
            print(chromosome.get_theta_list())
        print('\n')

    def print_circuits(self):
        """Prints all the generation's chromosome's circuits."""
        print("Circuits: ")
        for chromosome in self.chromosome_list:
            circuit = Circuit(chromosome)
            circuit.generate_circuit()
            circuit.draw()
        print("\n")

    def print_fitness(self):
        """Prints the generation's chromosome's fitness"""
        for fitness in self.fitness_list:
            print(fitness)
        print("\n")


class Circuit(object):
    """
    A qiskit QuantumCircuit made from a chromosome.

    Attributes
    ----------
    chromosome (Chromosome):
        The integer string representation of the _circuit.
    _circuit (Qiskit.QuantumCircuit):
        Qiskit representation of the chromosome. A _circuit that can be run and simulated.
    _SHOTS (int):
        Number of runs in the qiskit quantum _circuit simulator.
    _STARTING_STATES (List[list]):
        Possible cellular automata initial conditions for the 1D Von Neumann neighborhood.
    """

    def __init__(self, chromosome: Chromosome):
        """
        Circuit constructor. Takes a chromosome as parameter, and creates a Qiskit
        QuantumCircuit object form it.

        Parameters
        ----------
        chromosome: (Chromosome)
            The chromosome that describes the QuantumCircuit.
        """
        self.chromosome = chromosome
        self.circuit = QuantumCircuit(3, 1)
        self.shots = 1000
        self.STARTING_STATES = [[0, 0, 0],
                                [0, 0, 1],
                                [0, 1, 0],
                                [0, 1, 1],
                                [1, 0, 0],
                                [1, 0, 1],
                                [1, 1, 0],
                                [1, 1, 1]]
        self.results = {}

    def __repr__(self):
        """Returns a string visualizing the quantum _circuit"""
        return self.draw()

    def generate_circuit(self) -> None:
        """
        Parses the chromosome, and generates a Qiskit QuantumCircuit from it.
        """
        gates = int(self.chromosome.get_length() / 3)

        gate_dict = self.chromosome.get_gate_dict()

        for i in range(0, gates):
            gate_index = i * 3

            a = self.chromosome.get_integer_list()[gate_index]
            b = self.chromosome.get_integer_list()[gate_index + 1]
            c = self.chromosome.get_integer_list()[gate_index + 2]

            gate = gate_dict[str(a)]

            if gate == 'h':
                self.circuit.h(b)
            elif gate == 'cx':
                self.circuit.cx(b, c)
            elif gate == 'x':
                self.circuit.x(b)
            elif gate == 'swap':
                self.circuit.swap(b, c)
            elif gate == 'rzz':
                theta = self.chromosome.get_theta_list()[i]
                self.circuit.rzz(theta=theta, qubit1=b, qubit2=c)
            elif gate == 'rxx':
                theta = self.chromosome.get_theta_list()[i]
                self.circuit.rxx(theta=theta, qubit1=b, qubit2=c)
            elif gate == 'toffoli':
                target = b
                if target == 1:
                    self.circuit.toffoli(0, 2, target)
                else:
                    self.circuit.toffoli(abs(target - 1), abs(target - 2), target)
            elif gate == 'y':
                self.circuit.y(b)
            elif gate == 'z':
                self.circuit.z(b)
            else:
                print(gate + " is not a valid gate!")

        self.circuit.measure(0, 0)

    def calculate_probability_of_one(self) -> float:
        """Returns the measured chance of one after simulation"""
        counts = self.run_simulator()
        if '1' in counts:
            chance_of_one = counts['1'] / self.shots
        else:
            chance_of_one = 0.0

        return chance_of_one

    def find_chromosome_fitness(self, desired_chance_of_one: List[float]) -> float:
        """
        Calculates and return the fitness for the chromosome i.e.
        the sum of differences in all initial states.

        Parameters
        ----------
        desired_chance_of_one: List[float]
            A list of desired probabilities for all the CA initial states.

        Returns
        -------
        fitness: (float)
            The chromosome fitness.
        """
        fitness = 0
        for i in range(0, len(self.STARTING_STATES)):

            state = self.STARTING_STATES[i]
            probability = desired_chance_of_one[i]
            found_probability = self.find_init_state_probability(state)
            difference = abs(probability - found_probability)
            fitness = fitness + difference

        return fitness

    def find_kullback_liebler_fitness(self, desired_chance_of_one: List[float]) -> float:
        """
        Calculates and return the fitness for the chromosome with relative entropy (Kullback-Liebler).

        Parameters
        ----------
        desired_chance_of_one: List[float]
            A list of desired probabilities for all the CA initial states.

        Returns
        -------
        fitness: (float)
            The chromosome fitness.

        """
        fitness = 0
        probabilities = []
        found_probabilities = []
        for i in range(0, len(self.STARTING_STATES)):

            state = self.STARTING_STATES[i]
            found_probabilities.append(self.find_init_state_probability(state))
            probabilities.append(desired_chance_of_one[i])

        p = found_probabilities
        q = probabilities

        if p == 0:
            p = 0.00001
        if q == 0:
            q = 0.00001

        d = sum(rel_entr(p, q))

        fitness = fitness + d
        return fitness

    def find_init_state_probability(self, state: List[int]) -> float:
        """
        Finds the difference between the desired probability and measured probability for given state.

        Parameters
        ---------
        state: (List[int])
            A list with a binary triplet describing the initial state.

        Returns
        -------
        difference: (float)
            The difference between the desired probability and measured probability for given state.
        """
        self.clear_circuit()
        self.initialize_initial_states(state)
        self.generate_circuit()

        chance_of_one = self.calculate_probability_of_one()
        return chance_of_one

    def print_ca_outcomes(self, desired_chance_of_one: List[float]):
        """Prints a table of the results from a run of the chromosome"""
        print("Initial State | Desired outcome | Actual outcome  | Difference")
        total_diff = 0
        for i in range(0, len(self.STARTING_STATES)):
            state = self.STARTING_STATES[i]
            probability = desired_chance_of_one[i]

            found_probability = self.find_init_state_probability(state)
            # found_probability = self.find_kullback_liebler_fitness(state)
            difference = abs(found_probability - probability)
            total_diff = total_diff + difference
            desired_format = "{:.4f}".format(desired_chance_of_one[i])
            chance_format = "{:.4f}".format(found_probability)
            diff_format = "{:.4f}".format(difference)

            print(str(self.STARTING_STATES[i]) + "           "
                  + desired_format + "           "
                  + chance_format + "           "
                  + diff_format)
            self.clear_circuit()
        print("Total difference: " + str(total_diff))

    def get_total_difference(self, desired_chance_of_one: List[float]):
        total_diff = 0
        for i in range(0, len(self.STARTING_STATES)):
            state = self.STARTING_STATES[i]
            probability = desired_chance_of_one[i]

            found_probability = self.find_init_state_probability(state)
            difference = abs(found_probability - probability)
            total_diff = total_diff + difference

        return total_diff

    def print_counts(self):
        """Prints the counts result from simulation"""
        index = 0
        for triplet in self.STARTING_STATES:
            self.clear_circuit()
            self.initialize_initial_states(triplet)
            self.generate_circuit()
            print(self.run_simulator())
            index = index + 1

    def initialize_initial_states(self, triplet: List[int]) -> None:
        """
        Initializes a Cellular Automata (CA) state in the _circuit.

        Parameters
        ----------
        triplet: List[int]
            A list of three integer in {0,1} representing the one of the
            eight starting possibilities in 1D Von Neumann CA.
        """

        if triplet[0] == 1:
            self.circuit.x(0)
        if triplet[1] == 1:
            self.circuit.x(1)
        if triplet[2] == 1:
            self.circuit.x(2)

    def run_simulator(self) -> dict:
        """
        Runs the _circuit on the Qiskit AER simulator and returns the results as a dictionary.

        Returns
        -------
        counts: dict
            The results from the AER simulation.
        """
        aer_sim = Aer.get_backend('aer_simulator')
        # aer_sim = Aer.get_backend('aer_simulator_density_matrix')
        # aer_sim = Aer.get_backend('aer_simulator_stabilizer')
        quantum_circuit = assemble(self.circuit, shots=self.shots)
        job = aer_sim.run(quantum_circuit)
        counts = job.result().get_counts()
        return counts

    def draw(self) -> None:
        """Prints a visual representation of the _circuit"""
        print(self.circuit.draw(output='text'))

    def clear_circuit(self) -> None:
        """Clears the Qiskit QuantumCircuit for all gates"""
        self.circuit.data.clear()
