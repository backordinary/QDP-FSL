# https://github.com/vivekkatial/qaoa-three-sat/blob/c323ab9129707be553ab465a748c1f47aa315aaa/qaoa_three_sat/instance/three_sat.py
"""
This code contains the main Instance Class for QAOA 3SAT

Author: Vivek Katial
"""

# External Modules
import numpy as np
import pandas as pd
from qiskit import Aer, IBMQ
from qiskit import QuantumCircuit, execute
import mlflow

# Custom Modules
from qaoa_three_sat.rotation.rotations import Rotations
from qaoa_three_sat.utils.qc_helpers import calculate_rotation_angle_theta
from qaoa_three_sat.optimiser.nelder_mead import NelderMead
from qaoa_three_sat.optimiser.cma_es import CMA_ES
from qaoa_three_sat.optimiser.bfgs import BFGS


class QAOAInstance3SAT:
    """This class is a generic class for an instance of QAOA 3SAT and
    it consists of all the base methods needed for our circuit evolution

    Attributes
    ----------
        n_qubits : int
            The number of qubits
        n_rounds : int
            The number of rounds the QAOA circuit is built for
        single_rotations : list
            `Rotations` Object for Single Qubit Rotations
        double_rotations : list
            `Rotations` Object for Double Qubit Rotations
        triple_rotations : list
            `Rotations` Object for Triple Qubit Rotations
        classical_opt_alg : str
            The classical optimisation algorithm being utilised to calc angles
        optimiser_opts : dict
            A dictionary with optimisation algorithm parameters
        optimiser : object
            Optimiser Object
        classical_iter : int
            Number of iterations done on the classical optimisation algorithm
        circuit_init : bool
            Boolean on whether or not circuit initiated
        alpha : list
            A list of angle values alpha
        beta : list
            A list of angle values for beta
        backend : object
            An object representing where the simulation will run (e.g. `Aer.get_backend('statevector_simulator'))
        statevector : list
            An array of complex numbers representing the 2^n state vector
        hamiltonian : np.array()
            A `numpy` array containing the problem Hamiltonian
        quantum_circuit : object
            A `qiskit` quantum circuit object
        energy : float
            The energy cost function value being minimised
        pdf : list
            An array representing a probability distribution across all possible states
        sat_assgn : str
            A string representing the binary string for the satisfying assignment
        track_optimiser : bool
            A boolean on whether or not tracking of the classical optimizer should be enabled
        d_alpha : list
            An array containing the alpha angle settings at each iteration e.g. ``[[a_0, a_1], ... ]``
        d_beta : list
            An array containing the beta angle settings at each iteration e.g. ``[[b_0, b_1], ... ]``
        d_energy : list
            An array containing the energy angle settings at each iteration e.g. ``[e_1, ..., e_n]``
        d_instance : pandas.DataFrame()
            A pandas dataframe object with data
        disp : bool
            Set to True to print convergence messages.
        mlflow : bool
            Set to True to track results on the MLFlow Server
    """

    def __init__(
        self,
        n_qubits,
        n_rounds,
        single_rotations,
        double_rotations,
        triple_rotations,
        alpha,
        beta,
        classical_opt_alg,
        optimiser_opts,
        sat_assgn,
        track_optimiser=False,
        disp=False,
        mlflow=False,
        backend=Aer.get_backend("statevector_simulator"),
    ):

        self.n_qubits = n_qubits
        self.single_rotations = single_rotations
        self.double_rotations = double_rotations
        self.triple_rotations = triple_rotations

        # Optimization settings
        self.classical_opt_alg = classical_opt_alg
        self.optimiser_opts = optimiser_opts
        self.optimiser = None
        self.classical_iter = 0
        self.track_optimiser = track_optimiser
        self.sat_assgn = sat_assgn

        # Tracking attributes
        self.d_alpha = []
        self.d_beta = []
        self.d_energy = []
        self.disp = disp
        self.mlflow = mlflow

        # Quantum SubRoutine Settings
        self.n_rounds = n_rounds
        self.circuit_init = False
        self.alpha = alpha
        self.beta = beta
        self.backend = backend
        self.statevector = None
        self.pdf = None
        self.hamiltonian = None

        # Metric settings
        self.quantum_circuit = None
        self.d_instance = None
        self.energy = 0

    @property
    def n_qubits(self):
        """ Get n_qubits"""
        return self._n_qubits

    @n_qubits.setter
    def n_qubits(self, value):
        if not isinstance(value, int):
            raise TypeError("n_qubits must be an integer")
        if value <= 0:
            raise ValueError("n_qubits must be positive")
        self._n_qubits = value

    @property
    def single_rotations(self):
        """ Get Single Qubit Rotations"""
        return self._single_rotations

    @single_rotations.setter
    def single_rotations(self, value):
        if not isinstance(value, Rotations):
            raise TypeError("single_rotations must be a Rotations Object")
        self._single_rotations = value

    @property
    def double_rotations(self):
        """ Get Double Qubit Rotations"""
        return self._double_rotations

    @double_rotations.setter
    def double_rotations(self, value):
        if not isinstance(value, Rotations):
            raise TypeError("double_rotations must be a Rotations Object")
        self._double_rotations = value

    @property
    def triple_rotations(self):
        """ Get Triple Qubit Rotations"""
        return self._triple_rotations

    @triple_rotations.setter
    def triple_rotations(self, value):
        if not isinstance(value, Rotations):
            raise TypeError("triple_rotations must be a Rotations Object")
        self._triple_rotations = value

    @property
    def alpha(self):
        """ Get Alpha Rotation Angles"""
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        if not isinstance(value, list):
            raise TypeError("alpha must be a list")
        if len(value) != self.n_rounds:
            raise ValueError(
                "Incorrect number of alpha parameters passed. Should be %s - User passed %s"
                % (self.n_rounds, len(value))
            )
        self._alpha = value

    @property
    def beta(self):
        """ Get Beta Rotation Angles"""
        return self._beta

    @beta.setter
    def beta(self, value):
        if not isinstance(value, list):
            raise TypeError("beta must be a list")
        if len(value) != self.n_rounds:
            raise ValueError(
                "Incorrect number of beta parameters passed. Should be %s - User passed %s"
                % (self.n_rounds, len(value))
            )
        self._beta = value

    @property
    def classical_opt_alg(self):
        """ Get Classical Optimisation Algo"""
        return self._classical_opt_alg

    @classical_opt_alg.setter
    def classical_opt_alg(self, value):
        if not isinstance(value, str):
            raise TypeError("classical_opt_alg must be a string")
        # Add optimisers here as implemented
        if value not in ["nelder-mead", "cma-es", "bfgs"]:
            raise ValueError("classical_opt_alg currently not implemented")
        self._classical_opt_alg = value

    @property
    def optimiser_opts(self):
        """ Get Optimiser Option"""
        return self._optimiser_opts

    @optimiser_opts.setter
    def optimiser_opts(self, value):
        if not isinstance(value, dict):
            raise TypeError("optimiser_opts must be a `dict()`")
        if value["classical_opt_alg"] != self.classical_opt_alg:
            raise ValueError(
                "Optimiser options not consistent with Optimiser:%s"
                % self.classical_opt_alg
            )
        self._optimiser_opts = value

    @property
    def track_optimiser(self):
        """Get Tracking Optimiser Bool"""
        return self._track_optimiser

    @track_optimiser.setter
    def track_optimiser(self, value):
        if not isinstance(value, bool):
            raise TypeError("track_optimiser must be a bool")
        self._track_optimiser = value

    @property
    def disp(self):
        """Get disp param bool"""
        return self._disp

    @disp.setter
    def disp(self, value):
        if not isinstance(value, bool):
            raise TypeError("disp must be a bool")
        self._disp = value

    @property
    def mlflow(self):
        return self._mlflow

    @mlflow.setter
    def mlflow(self, value):
        if not isinstance(value, bool):
            raise TypeError("mlflow must be a bool")
        self._mlflow = value

    @property
    def sat_assgn(self):
        return self._sat_assgn

    @sat_assgn.setter
    def sat_assgn(self, value):
        if not isinstance(value, str):
            raise TypeError("sat_assgn must be a string")
        self._sat_assgn = value

    def initiate_circuit(self):
        """A function to initiate circuit as a qiskit QC circuit object.
        ...
        :return: An initialised quantum circuit with hadmards across all qubits
        :rtype: qiskit.circuit.quantumcircuit.QuantumCircuit
        """
        self.circuit_init = True
        self.quantum_circuit = QuantumCircuit(self.n_qubits, self.n_qubits)
        self.quantum_circuit.h(range(self.n_qubits))
        self.quantum_circuit.barrier()
        return self.quantum_circuit

    def add_single_rotations(self, n_round):
        """ Adding single qubit rotations to circuit"""
        # Apply single qubit rotations
        for qubit in self.single_rotations.rotations:
            theta = calculate_rotation_angle_theta(
                self.alpha[n_round], qubit["coefficient"]
            )
            self.quantum_circuit.rz(theta, qubit["qubits"][0])

    def add_double_rotations(self, n_round):
        """ Adding two qubit rotations to circuit"""

        # Apply double qubit rotations
        for qubit in self.double_rotations.rotations:
            theta = calculate_rotation_angle_theta(
                self.alpha[n_round], qubit["coefficient"]
            )
            # Apply on Gate Z_i Z_j
            self.quantum_circuit.cx(qubit["qubits"][0], qubit["qubits"][1])
            self.quantum_circuit.rz(theta, qubit["qubits"][1])
            self.quantum_circuit.cx(qubit["qubits"][0], qubit["qubits"][1])

    def add_triple_rotations(self, n_round):
        """Adding three qubit rotation terms into circuit"""

        # Apply 3 qubit rotations
        for qubit in self.triple_rotations.rotations:
            theta = calculate_rotation_angle_theta(
                self.alpha[n_round], qubit["coefficient"]
            )
            self.quantum_circuit.cx(qubit["qubits"][0], qubit["qubits"][1])
            self.quantum_circuit.cx(qubit["qubits"][1], qubit["qubits"][2])
            self.quantum_circuit.rz(theta, qubit["qubits"][2])
            self.quantum_circuit.cx(qubit["qubits"][1], qubit["qubits"][2])
            self.quantum_circuit.cx(qubit["qubits"][0], qubit["qubits"][1])

    def close_round(self, n_round):
        """Closing the round of a quantum circuit (then measuring)

        :param n_round: Number of rounds to build the circuit for
        :type n_round: int
        """
        self.quantum_circuit.barrier()
        # Apply X rotations
        self.quantum_circuit.rx(self.beta[n_round], range(self.n_qubits))
        self.quantum_circuit.barrier()
        # self.quantum_circuit.measure(range(self.n_qubits), range(self.n_qubits))

    def build_circuit(self):
        """
        Class method to build the quantum circuit.
        """

        # Initiate Quantum Circuit
        self.initiate_circuit()

        for i in range(self.n_rounds):
            self.add_single_rotations(n_round=i)
            self.add_double_rotations(n_round=i)
            self.add_triple_rotations(n_round=i)
            self.close_round(i)

    def simulate_circuit(self):
        """
        Simulate the quantum circuit and get the corresponding statevector
        """
        self.statevector = (
            execute(self.quantum_circuit, self.backend).result().get_statevector()
        )

    def build_hamiltonian(self):
        """
        Calculate circuit energy
        """
        self.single_rotations.build_hamiltonian()
        self.double_rotations.build_hamiltonian()
        self.triple_rotations.build_hamiltonian()
        # Build circuit Hamiltonian
        self.hamiltonian = (
            self.single_rotations.hamiltonian
            + self.double_rotations.hamiltonian
            + self.triple_rotations.hamiltonian
        )

    def cost_function(self, angles):
        """Circuit Cost function, run the circuit and measure the energy

        :param angles: Angle theta found by classical optimiser
        :type angles: list
        :returns: self.energy
        :rtype: {float}
        """
        # Update angles
        if self.disp:
            print(
                "Classical Optimization Iteration %s: \t alpha=%s \t beta=%s \t energy=%s"
                % (
                    self.classical_iter,
                    angles[0 : self.n_rounds],
                    angles[self.n_rounds :],
                    self.energy,
                )
            )

        self.alpha = angles[0 : self.n_rounds].tolist()
        self.beta = angles[self.n_rounds :].tolist()

        # Rebuild Circuit
        self.quantum_circuit = None
        self.build_circuit()

        self.classical_iter += 1
        # Run circuit
        self.simulate_circuit()
        self.measure_energy()

        # If tracking enabled - collect data on alpha, beta
        self.d_alpha.append(self.alpha)
        self.d_beta.append(self.beta)
        self.d_energy.append(self.energy)

        return self.energy

    def measure_energy(self):
        """
        Calculate circuit energy
        """
        self.build_hamiltonian()
        ham_state = np.matmul(self.hamiltonian, self.statevector)
        energy = np.dot(ham_state, np.conjugate(self.statevector))
        self.energy = energy.real

    def optimise_circuit(self):
        """
        Method to optimise the circuit
        """

        if self.quantum_circuit is None:
            raise AttributeError("Please Build Circuit before Optimization")

        # Construct n-d array for Nelder-Mead
        angles = [self.alpha, self.beta]
        angles = [angle for i in angles for angle in i]

        # Build Optimiser Class
        if self.classical_opt_alg == "nelder-mead":
            # Initialise Nelder Mead
            self.optimiser = NelderMead(
                vars_vec=angles,
                cost_function=self.cost_function,
                options=self.optimiser_opts,
            )
        # Run CMA ES
        elif self.classical_opt_alg == "cma-es":
            # Initialise CMA ES
            self.optimiser = CMA_ES(
                vars_vec=angles,
                cost_function=self.cost_function,
                options=self.optimiser_opts,
            )

        elif self.classical_opt_alg == "bfgs":
            # Initialise CMA ES
            self.optimiser = BFGS(
                vars_vec=angles,
                cost_function=self.cost_function,
                options=self.optimiser_opts,
            )

        else:
            # Raise Error if a valid algorithm not specified
            raise ValueError(
                "Please Specify a valid Algorithm, %s is not implemented"
                % self.classical_opt_alg
            )

        # Optimise Instance & Circuit
        self.optimiser.optimise()
        self.generate_instance_df()

    def generate_instance_df(self):
        """A function to generate a dataframe containing columns for angles and the minimum energy"""

        # Create alpha key strings
        l_alpha = ["alpha_" + str(i) for i in range(len(self.alpha))]
        # Create beta key strings
        l_beta = ["beta_" + str(i) for i in range(len(self.alpha))]
        # Create all
        l_all = [l_alpha, l_beta]
        l_keys = [item for sublist in l_all for item in sublist]
        l_keys.append("energy")

        # Data dictionary
        d_dict = dict.fromkeys(l_keys)
        d_instance = []

        for row in zip(self.d_alpha, self.d_beta, self.d_energy):
            # initiate row element
            row_el = d_dict

            # Build alpha vals df
            for i, alpha in enumerate(l_alpha):
                row_el[alpha] = row[0][i]

            # Build beta vals df
            for i, beta in enumerate(l_beta):
                row_el[beta] = row[1][i]

            row_el["energy"] = row[2]

            d_instance.append(row_el)

        # Add optimisation stuff
        d_instance = pd.DataFrame(d_instance)
        d_instance["algorithm"] = self.classical_opt_alg
        d_instance["optimiser_opts"] = str(self.optimiser_opts)

        d_instance = d_instance.loc[[d_instance["energy"].idxmin()]]
        d_instance.to_csv("data/sample_test.csv")

        # Allocate data for instance
        self.d_instance = d_instance
        return 0

    def calculate_pdf(self):
        """
        This function generates a PDF from the instance state vector
        """
        self.pdf = np.multiply(self.statevector, np.conjugate(self.statevector))
