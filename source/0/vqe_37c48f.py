# https://github.com/AndersHR/qem__master_thesis/blob/b032a90b683558404a6408fc9570850400c8d12b/VQE.py
from qiskit import *
from qiskit.circuit import Instruction, Parameter, ParameterVector
from qiskit.chemistry import FermionicOperator
from qiskit.chemistry.drivers import PySCFDriver#, Molecule

#from qiskit_nature.problems.second_quantization import ElectronicStructureProblem
#from qiskit_nature.converters.second_quantization import QubitConverter
#from qiskit_nature.mappers.second_quantization import JordanWignerMapper
#from qiskit_nature.drivers import PySCFDriver, Molecule, UnitsType

from qiskit.transpiler import PassManager, CouplingMap, Layout
from qiskit.transpiler.passes import ApplyLayout, SetLayout, LookaheadSwap, TrivialLayout, Optimize1qGates

from qiskit.aqua.algorithms import NumPyEigensolver
from qiskit.aqua.components.optimizers import SPSA

from qiskit.result.result import ExperimentResult, Result

from qiskit.quantum_info.operators import Pauli

from qiskit.aqua.operators.legacy import WeightedPauliOperator

import numpy as np
import pickle, os, sys
from dataclasses import dataclass
from typing import *

from scipy.optimize import minimize

sys.path.append('../')

#from error_mitigation.zero_noise_extrapolation import Richardson_extrapolate
from error_mitigation.zero_noise_extrapolation_cnot import noise_amplify_cnots

def hex_to_binstring(hex_str: str, n_bits: int = None) -> bin:
    if n_bits is None:
        return bin(int(hex_str, 16))[2::]
    else:
        return bin(int(hex_str, 16))[2::].zfill(n_bits)

def default_decision_rule(mmt_str: str):
    """
    Takes in a measurement counts string, e.g. mmt_str = "01101", from measurements on error detection ancillas.
    Return TRUE if the measurement results corresponds to an error being detected, and FALSE otherwise

    :param mmt_str:
    :return:
    """
    return mmt_str == "1"

# VQE functions

def get_qubit_operator(geometry, multiplicity: int = 1, basis: str = "sto-3g", map_type: str = "jordan_wigner") \
                      -> (WeightedPauliOperator, float):
    """
    
    :param geometry: List[Tuple[str,list]]
        Molecule geometry, e.g., [("H",[0,0,0]), ("H,[0,0,0.74])]
    :param multiplicity: int
        Molecule multiplicity
    :param basis: str
        Basis
    :param map_type: str
        The fermionic-to-qubit operator mapping, e.g., "jordan_wigner" / "parity" / "bravyi_kitaev" / "bksf"
    :return: 
    """
    #molecule = Molecule(geometry=geometry, multiplicity=multiplicity)
    driver = PySCFDriver(atom=geometry, basis=basis)
    molecule = driver.run()

    energy_shift = molecule.nuclear_repulsion_energy

    #es_problem = ElectronicStructureProblem(driver)
    #second_q_ops = es_problem.second_q_ops()

    #qubit_converter = QubitConverter(mapper=JordanWignerMapper())
    #qubit_op = qubit_converter.convert(second_q_ops[0])
    fermionic_operator = FermionicOperator(h1=molecule.one_body_integrals, h2=molecule.two_body_integrals)

    qubit_operator = fermionic_operator.mapping(map_type=map_type)

    return qubit_operator, energy_shift

def is_pure_pauli_op(composite_pauli_op: str, pauli_check_type: str = "Z"):
    for op in composite_pauli_op:
        if op != pauli_check_type and op != "I":
            return False
    return True


# Dataclasses specifically for containing information efficiently when reading/writing to/from files

@dataclass(frozen=True)
class VQEOperator:
    qubit_operator: WeightedPauliOperator
    nuclear_repulsion_energy: float


@dataclass(frozen=True)
class EnergyResult:
    energy: float
    noise_amplified_energies: np.ndarray
    variances: np.ndarray
    discarded_rates: np.ndarray
    shots: int
    params: np.ndarray
    # Error controlled sampling variables
    #error_controlled_sampling: bool
    #min_shots: int
    #max_shots: int


@dataclass(frozen=True)
class NoiseAmplifiedEnergyResult:
    energy: float
    amp_factor: int
    variance: float
    discarded_rate: float
    shots: int
    params: np.ndarray
    #performed_shots: np.ndarray


@dataclass(frozen=True)
class VQEResult:
    energy: float
    params: np.ndarray
    #x0: np.ndarray
    #iterations: int
    #hamiltonian: dict


# VQE main class

class VQE:

    def __init__(self, geometry, ansatz: QuantumCircuit, parameters: Union[List[Parameter], ParameterVector],
                 multiplicity: int = 1, basis: str = "sto-3g", map_type: str = "jordan_wigner",
                 mmt_qubits = None, backend = None, noise_model = None, shots: int = 8192,
                 error_detect_qubits: List[int] = None, error_detect_qc: QuantumCircuit = None,
                 decision_rule: Callable = None, n_amp_factors: int = 1,
                 save_results: bool = False, experiment_name: str = "", directory: str = "results",
                 save_op: bool = False, error_controlled_sampling: bool = True, error_tol: float = 0.01,
                 max_shots: int = 100*8192,
                 ):

        # Saving / writing of results
        self.save_results = save_results
        self.experiment_name = experiment_name
        self.directory = directory

        self.save_op = save_op

        # Molecule and Hamiltonian transformation parameters
        self.geometry = geometry
        self.multiplicity = multiplicity

        self.basis = basis
        self.map_type = map_type

        # Get the transformed Hamiltonian operator and energy shift from nuclear repulsion energy
        self.qubit_operator, self.nuclear_repulsion_energy = self.get_qubit_op()
        self.energy_shift = self.nuclear_repulsion_energy

        # Number of qubits for molecular state representations
        self.n_mol_qubits = self.qubit_operator.num_qubits

        self.ansatz = ansatz.copy()
        self.tot_qubits, self.tot_clbits = self.ansatz.num_qubits, self.ansatz.num_clbits
        self.parameters = parameters

        if mmt_qubits is None:
            self.mmt_qubits = [i for i in range(self.n_mol_qubits)]
        else:
            if len(mmt_qubits) != self.n_mol_qubits:
                raise Exception("mmt_qubits was of length {:}, but should be {:}.".format(len(mmt_qubits),
                                                                                          self.n_mol_qubits))
            self.mmt_qubits = mmt_qubits

        if error_detect_qc is None:
            self.error_detect_qc = None
        else:
            self.error_detect_qc = error_detect_qc.copy()
        self.error_detect_qubits = error_detect_qubits
        if error_detect_qubits is None or error_detect_qc is None:
            self.error_detect = False
        else:
            self.error_detect = True

        if decision_rule is None:
            self.decision_rule = default_decision_rule
        else:
            self.decision_rule = decision_rule

        self.discarded = {}

        # Validation of circuits
        if not self.error_detect:
            if self.n_mol_qubits > self.tot_qubits:
                raise Exception("Ansatz circuit should contain at least {:} qubits.".format(self.n_mol_qubits))
            elif self.n_mol_qubits > self.tot_clbits:
                raise Exception("Ansatz circuit should contain at least {:} classical bits.".format(self.n_mol_qubits))
        elif self.error_detect:
            if self.n_mol_qubits >= self.tot_qubits:
                raise Exception("Ansatz circuit should contain at least {:} qubits.".format(self.n_mol_qubits + 1))
            elif self.n_mol_qubits >= self.tot_clbits:
                raise Exception("Ansatz circuit should contain at least {:} classical bits.".format(self.n_mol_qubits + 1))

            if self.error_detect_qc.num_qubits != self.tot_qubits or self.error_detect_qc.num_clbits != self.tot_clbits:
                raise Exception("Error detection circuit " +
                                "(num_qubits={:}, num_clbits={:}) ".format(self.error_detect_qc.num_clbits,
                                                                           self.error_detect_qc.num_clbits) +
                                 "incompatible with ansatz circuit " +
                                 "(num_qubits={:}, num_clbits={:})".format(self.tot_qubits, self.tot_clbits))


        # Variables involved in circuit executions

        if backend is None:
            self.backend = Aer.get_backend("qasm_simulator")
        else:
            self.backend = backend

        self.noise_model = noise_model
        self.shots = shots

        self.error_controlled_sampling = error_controlled_sampling
        self.error_tol = error_tol
        self.max_shots = max_shots

        # Zero noise extrapolation variables

        self.n_amp_factors = n_amp_factors

        self.amplification_factors = np.asarray([2*i + 1 for i in range(n_amp_factors)])
        self.gamma_coeffs = self.get_richardson_coefficients()

        # Initialize further variables for later use

        self.hamiltonian_dict = {}
        self.build_hamiltonian_dict()

        # Transpile the ansatz circuit and combine circuits

        self.measurement_circuits = self.build_parameterized_circuits()

        # Define variables for later use

        self.qc = None
        self.layout = None
        self.counts = None

        self.verbose = False
        self.iter = 0

        self.discarded_rates = None
        self.noise_mitigated_energies = None
        self.variances = None
        self.performed_shots = None
        self.energy, self.optimal_params = None, None

    def set_geometry(self, geometry):
        self.geometry = geometry
        self.qubit_operator, self.nuclear_repulsion_energy = self.get_qubit_op()

    def get_qubit_op(self) -> (WeightedPauliOperator, float):
        filename = self.directory + "/" + self.experiment_name
        filename += "_multiplicity{:}".format(self.multiplicity) + "_basis{:}".format(self.basis)
        filename += "_map_type{:}".format(self.map_type)

        if self.save_op:
            op = self.read_from_file(filename)

            if not ((op is None) or (type(op) is not VQEOperator)):
                operator_loaded = True

                qubit_op, nuclear_repulsion_energy = op.qubit_operator, op.nuclear_repulsion_energy
                return qubit_op, nuclear_repulsion_energy

        qubit_op, nuclear_repulsion_energy = get_qubit_operator(geometry=self.geometry, multiplicity=self.multiplicity,
                                                                basis=self.basis, map_type=self.map_type)

        if self.save_op:
            op = VQEOperator(qubit_operator=qubit_op, nuclear_repulsion_energy=nuclear_repulsion_energy)

            self.write_to_file(filename, op)

        return qubit_op, nuclear_repulsion_energy

    def get_richardson_coefficients(self):
        if self.n_amp_factors == 1:
            return np.asarray([1])

        A = np.zeros((self.n_amp_factors, self.n_amp_factors))
        b = np.zeros((self.n_amp_factors, 1))

        A[0,:] = 1
        b[0] = 1

        for k in range(1, self.n_amp_factors):
            A[k,:] = self.amplification_factors**k

        gamma_coeffs = np.linalg.solve(A,b)

        return gamma_coeffs

    def richardson_extrapolate(self, noise_amplified_energies):
        if self.n_amp_factors == 1:
            return noise_amplified_energies[0]
        return np.dot(np.transpose(noise_amplified_energies), self.gamma_coeffs)[0]

    # FUNCTIONS FOR READING / WRITING RESULTS FROM / TO FILES

    def get_filename_base(self, mode: str, backend_name: str, shots: int, method: Union[str, None],
                          tol: Union[float, None], n_amp_factors: int, error_controlled_sampling: bool,
                          error_tol: float, max_shots: int, end: str, directory: str = "results"):
        filename = self.experiment_name

        filename += "_{:}_backend{:}".format(mode, backend_name)

        filename += "_errcontr{:}".format(str(error_controlled_sampling))

        if error_controlled_sampling:
            filename += "_sampleshots{:}_errortol{:}_maxshots{:}".format(str(shots),str(error_tol), str(max_shots))
        else:
            filename += "_shots{:}".format(str(shots))

        if not method is None:
            filename += "_optmethod{:}_tol{:}".format(method,str(tol))

        filename += "_errdetect{:}".format(str(self.error_detect))

        if n_amp_factors == 1:
            filename += "_zneFalse"
        else:
            filename += "_zneTrue_nampfactors{:}".format(n_amp_factors)

        return directory + "/" + filename + "." + end

    def get_vqe_filename(self, backend_name: str, shots: int, method: str, tol: float, n_amp_factors: int,
                         error_controlled_sampling: bool, error_tol: float, max_shots: int,
                         directory: str = "results"):
        return self.get_filename_base("VQE", backend_name, shots, method, tol, n_amp_factors, error_controlled_sampling,
                                      error_tol, max_shots, "result", directory)

    def get_energycomputation_filename(self, backend_name: str, shots: int, n_amp_factors: int,
                                       error_controlled_sampling: bool, error_tol: float, max_shots: int,
                                       directory: str = "results"):
        return self.get_filename_base("ENERGY", backend_name, shots, None, None, n_amp_factors, error_controlled_sampling,
                                      error_tol, max_shots, "energycomp", directory)

    def get_noiseamplifiedenergy_filename(self, backend_name: str, shots: int, n_amp_factors: int,
                                       error_controlled_sampling: bool, error_tol: float, max_shots: int,
                                       amp_factor: int, directory: str = "results"):
        return self.get_filename_base("NAMP_ENERGY", backend_name, shots, None, None, n_amp_factors, error_controlled_sampling,
                                      error_tol, max_shots, "ampenergyr{:}".format(amp_factor), directory)

    @staticmethod
    def read_from_file(filename: str):
        if os.path.isfile(filename):
            file = open(filename, "rb")
            data = pickle.load(file)
            file.close()
            return data
        else:
            return None

    @staticmethod
    def write_to_file(filename: str, data: Any):
        file = open(filename, "wb")
        pickle.dump(data, file)
        file.close()

    def read_result(self, backend_name: str, shots: int, method: str, tol: float, n_amp_factors: int,
                    error_controlled_sampling: bool, error_tol: float, max_shots: int,):
        filename = self.get_vqe_filename(backend_name, shots, method, tol, n_amp_factors,
                                         error_controlled_sampling, error_tol, max_shots, directory="results")

        res = self.read_from_file(filename)

        if (res is None) or (type(res) is not VQEResult):
            return None, None

        energy, optimal_params = res.energy, res.params

        return energy, optimal_params

    def write_result(self, energy, optimal_params, backend_name: str, shots: int, method: str, tol: float,
                     n_amp_factors: int, error_controlled_sampling: bool, error_tol: float, max_shots: int,):
        filename = self.get_vqe_filename(backend_name, shots, method, tol, n_amp_factors,
                                         error_controlled_sampling, error_tol, max_shots, directory="results")

        res = VQEResult(energy=energy, params=optimal_params)

        self.write_to_file(filename, data=res)

    def read_energycomputation_result(self, params: Union[list, np.ndarray], backend_name: str, shots: int,
                                      n_amp_factors: int, error_controlled_sampling: bool, error_tol: float,
                                      max_shots: int, directory: str = "results"):
        filename = self.get_energycomputation_filename(backend_name, shots, n_amp_factors, error_controlled_sampling,
                                                       error_tol, max_shots, directory=directory)

        res = self.read_from_file(filename)

        if (res is None) or (type(res) is not EnergyResult):
            return None, None, None, None, None

        loaded_params = res.params

        if not np.array_equal(params, loaded_params):
            return None, None, None, None, None

        energy = res.energy
        noise_amplified_energies = res.noise_amplified_energies
        variances = res.variances
        performed_shots = res.shots
        discarded_rates = res.discarded_rates

        return energy, noise_amplified_energies, variances, discarded_rates, performed_shots

    def write_energycomputation_result(self, energy, noise_amplified_energies, variances, discarded_rates, performed_shots,
                                       params: Union[list, np.ndarray], backend_name: str, shots: int,
                                       n_amp_factors: int, error_controlled_sampling: bool, error_tol: float,
                                       max_shots: int, directory: str = "results"):

        filename = self.get_energycomputation_filename(backend_name, shots, n_amp_factors, error_controlled_sampling,
                                                       error_tol, max_shots, directory=directory)

        res = EnergyResult(energy=energy, noise_amplified_energies=noise_amplified_energies, variances=variances,
                           discarded_rates=discarded_rates, shots=performed_shots, params=params)

        self.write_to_file(filename, data=res)

    def read_noiseamplified_result(self, params: Union[list, np.ndarray], backend_name: str, shots: int, n_amp_factors: int,
                                   error_controlled_sampling: bool, error_tol: float, max_shots: int,
                                   amp_factor: int, directory: str = "results"):
        filename = self.get_noiseamplifiedenergy_filename(backend_name, shots, n_amp_factors, error_controlled_sampling,
                                                          error_tol, max_shots, amp_factor, directory=directory)

        res = self.read_from_file(filename)

        if (res is None) or (type(res) is not NoiseAmplifiedEnergyResult):
            return None, None, None, None

        loaded_params = res.params

        if not np.array_equal(params, loaded_params):
            return None, None, None, None

        noise_amplified_energy = res.energy
        variance = res.variance
        discarded_rate = res.discarded_rate
        performed_shots = res.shots

        return noise_amplified_energy, variance, discarded_rate, performed_shots

    def write_noiseamplified_result(self, noise_amplified_energy, variance, discarded_rate, performed_shots,
                                    params: Union[list, np.ndarray], backend_name: str, shots: int, n_amp_factors: int,
                                    error_controlled_sampling: bool, error_tol: float, max_shots: int,
                                    amp_factor: int, directory: str = "results"):
        filename = self.get_noiseamplifiedenergy_filename(backend_name, shots, n_amp_factors, error_controlled_sampling,
                                                          error_tol, max_shots, amp_factor, directory=directory)

        res = NoiseAmplifiedEnergyResult(energy=noise_amplified_energy, variance=variance, discarded_rate=discarded_rate,
                                         shots=performed_shots, amp_factor=amp_factor, params=params)

        self.write_to_file(filename, data=res)

    # SET UP DICT OVER HAMILTONIAN OPERATORS

    def build_hamiltonian_dict_entry(self, operator_key, coeff, pauli_str):
        if operator_key == "IIII":
            return
        if operator_key not in self.hamiltonian_dict.keys():
            self.hamiltonian_dict[operator_key] = {"terms": {}}
        self.hamiltonian_dict[operator_key]["terms"][pauli_str] = np.real(coeff)

    def build_hamiltonian_dict(self):
        self.hamiltonian_dict = {}
        for (coeff, pauli_op) in self.qubit_operator.paulis:
            pauli_str = pauli_op.to_label()
            if pauli_str == "IIII":
                self.energy_shift += np.real(coeff)
            elif is_pure_pauli_op(pauli_str, "Z"):
                self.build_hamiltonian_dict_entry("ZZZZ", coeff, pauli_str)
            elif is_pure_pauli_op(pauli_str, "X"):
                self.build_hamiltonian_dict_entry("XXXX", coeff, pauli_str)
            elif is_pure_pauli_op(pauli_str, "Y"):
                self.build_hamiltonian_dict_entry("YYYY", coeff, pauli_str)
            else:
                self.build_hamiltonian_dict_entry(pauli_str, coeff, pauli_str)

    # QUANTUM CIRCUIT HELP FUNCTIONS

    def callback_get_layout(self, **kwargs):
        self.layout = kwargs["property_set"]["layout"]

    def apply_layout(self, qc, layout=None):
        if layout is None:
            layout = self.layout
        if self.layout is None:
            layout = Layout.from_qubit_list(self.ansatz.qubits)

        set_layout_pass = SetLayout(layout=layout)
        apply_layout_pass = ApplyLayout()

        pm = PassManager([set_layout_pass, apply_layout_pass])

        return pm.run(qc)

    def noise_amplify(self, qc: QuantumCircuit, amp_factor: int):
        noise_amplified_qc = noise_amplify_cnots(qc=qc.copy(), amp_factor=amp_factor)
        return noise_amplified_qc

    # FUNCTIONS FOR CONSTRUCTION AND EXECUTIONS OF QUANTUM CIRCUITS

    def build_measurement_subcircuit(self, pauli_str: str) -> QuantumCircuit:

        qc_subcircuit = QuantumCircuit(self.tot_qubits, self.tot_clbits)

        qc_subcircuit.barrier()

        for i, pauli_char in enumerate(pauli_str):
            if pauli_char == "X":
                qc_subcircuit.h(self.mmt_qubits[i])
                qc_subcircuit.measure(self.mmt_qubits[i], self.mmt_qubits[i])
            elif pauli_char == "Y":
                qc_subcircuit.sdg(self.mmt_qubits[i])
                qc_subcircuit.h(self.mmt_qubits[i])
                qc_subcircuit.measure(self.mmt_qubits[i], self.mmt_qubits[i])
            else:
                qc_subcircuit.measure(self.mmt_qubits[i], self.mmt_qubits[i])

        return qc_subcircuit

    def build_parameterized_circuits(self):
        if self.backend.configuration().simulator:
            trivial_layout = Layout.from_qubit_list(self.ansatz.qubits)
            pm = PassManager([SetLayout(layout=trivial_layout), ApplyLayout()])
            self.ansatz = pm.run(self.ansatz)

        if self.error_detect:
            self.ansatz = self.ansatz.compose(self.error_detect_qc)

        self.ansatz = transpile(self.ansatz, backend=self.backend, optimization_level=3,
                                callback=self.callback_get_layout)

        mmt_circuits = {}
        for pauli_str in self.hamiltonian_dict.keys():
            mmt_qc = self.build_measurement_subcircuit(pauli_str=pauli_str)

            mmt_qc = self.apply_layout(mmt_qc)
            mmt_circuits[pauli_str] = mmt_qc

        return mmt_circuits

    def build_measurement_circuits(self, ansatz: QuantumCircuit):
        circuits = []
        for pauli_string in self.measurement_circuits.keys():
            qc = ansatz.compose(self.measurement_circuits[pauli_string])
            qc.name = pauli_string

            # TODO: Run Optimize1gGates pass

            circuits.append(qc.copy())
        return circuits

    def build_execution_circuits(self, qc: QuantumCircuit, shots: int = 8192) -> (List[QuantumCircuit], int, int):
        if shots <= 8192:
            partitioned_shots, repeats = 8192, 1
        elif shots % 8192 == 0:
            partitioned_shots, repeats = 8192, (shots // 8192)
        else:
            repeats = (shots // 8192) + 1
            partitioned_shots = int(shots / repeats)

        execution_circuits = [qc.copy() for i in range(repeats)]
        return execution_circuits, partitioned_shots

    # FUNCTIONS FOR COMPUTING ENERGY EXPECTATION VALUES AND OTHER RELATED MEASURES

    def compute_pauliterm_exp_val(self, results: List[ExperimentResult], pauli_str: str) -> (float, float):
        exp_vals, shots_kept, shots_discarded = np.zeros(len(results)), np.zeros(len(results)), np.zeros(len(results))
        for i, experiment_result in enumerate(results):
            counts = experiment_result.data.counts
            e, tot, discarded = 0, 0, 0

            for key in counts.keys():
                count_str = hex_to_binstring(key, self.tot_clbits)[::-1]

                eigenval = +1

                errors_detected = False
                if self.error_detect:
                    error_detect_str = ""
                    for error_qubit in self.error_detect_qubits:
                        error_detect_str += count_str[error_qubit]
                        errors_detected = self.decision_rule(error_detect_str)
                # If errors have been detected, discard the measurement result. Otherwise, go on with your life.
                if errors_detected:
                    discarded += counts[key]
                else:
                    for j, q in enumerate(self.mmt_qubits[::-1]):
                        if pauli_str[j] != "I":
                            if count_str[q] == "1":
                                eigenval = eigenval * -1
                    tot += counts[key]
                    e += eigenval * counts[key]
            exp_vals[i], shots_kept[i], shots_discarded[i] = e / tot, tot, discarded

        return np.average(exp_vals), np.average(shots_discarded / (shots_kept + shots_discarded))

    def compute_energy_from_pauli_term_exp_vals(self, pauli_term_exp_vals: dict, pauli_term_discarded_rates: dict = None)\
            -> (float, float, float):
        energy, variance, discarded = 0, 0, []

        for measurement_op in self.hamiltonian_dict.keys():
            for pauli_str in self.hamiltonian_dict[measurement_op]["terms"].keys():
                coeff = self.hamiltonian_dict[measurement_op]["terms"][pauli_str]
                exp_val, discarded_rate = pauli_term_exp_vals[pauli_str], pauli_term_discarded_rates[pauli_str]
                discarded.append(discarded_rate)

                # Compute energy contribution from Hamiltonian pauli term
                energy += coeff * exp_val

                # Compute variance contribution from Hamiltonian pauli term
                variance += (coeff**2) * (1 - exp_val**2)

        return energy + self.energy_shift, variance, np.average(discarded)

    def compute_energy_from_results(self, results_dict: dict) -> (float, float, float):
        pauli_term_exp_vals, pauli_term_discarded_rates = {}, {}
        for measurement_op in results_dict.keys():
            results = results_dict[measurement_op]
            if measurement_op == "Z"*len(measurement_op):
                self.counts = results[0].data.counts
            for pauli_str in self.hamiltonian_dict[measurement_op]["terms"].keys():
                exp_val, discarded_rate = self.compute_pauliterm_exp_val(results, pauli_str)
                pauli_term_exp_vals[pauli_str], pauli_term_discarded_rates[pauli_str] = exp_val, discarded_rate
        return self.compute_energy_from_pauli_term_exp_vals(pauli_term_exp_vals, pauli_term_discarded_rates)

    def execute_circuits(self, circuits: List[QuantumCircuit], shots: int = None) -> dict:
        if shots is None:
            shots = self.shots

        results_dict = {}
        for qc in circuits:
            execution_circuits, partitioned_shots = self.build_execution_circuits(qc, shots)

            if self.noise_model is None:
                job = execute(execution_circuits, backend=self.backend, shots=partitioned_shots,
                              optimization_level=0)
            else:
                job = execute(execution_circuits, backend=self.backend, shots=partitioned_shots,
                              noise_model=self.noise_model, optimization_level=0)

            results_dict[qc.name] = job.result().results

        return results_dict

    # COARSE SEARCH

    def coarse_search_randomsample(self, n: int = 20):
        # TODO: Implement coarse parameter search
        return


    # COMPLETE VQE FUNCTIONS

    def compute_energy_sample(self, noise_amplified_ansatz: QuantumCircuit, shots: int = 8192) -> (float, float, float, float):
        circuits = self.build_measurement_circuits(noise_amplified_ansatz)
        results_dict = self.execute_circuits(circuits, shots=shots)
        noise_amplified_energy, variance, discarded_rate = self.compute_energy_from_results(results_dict)
        return noise_amplified_energy, variance, discarded_rate

    def compute_noise_amplified_energy(self, ansatz: QuantumCircuit, error_tol: float, gamma_coeff: float, amp_factor: int = 1,
                                       sample_shots: int = 8192, max_shots: int = 100 * 8192, verbose: bool = False) \
            -> (float, float, float, float):

        noise_amplified_ansatz = self.noise_amplify(ansatz, amp_factor)

        energies, variances, discarded_rates = np.zeros(2), np.zeros(2), np.zeros(2)

        shot_samples = [sample_shots, None]

        # Do a first, initial sample
        energies[0], variances[0], discarded_rates[0] = self.compute_energy_sample(noise_amplified_ansatz, shot_samples[0])

        if not self.error_controlled_sampling:
            print("test1")
            return energies[0], variances[0], discarded_rates[0], shot_samples[0]

        if verbose:
            print("performing error controlled sampling, error tol =", self.error_tol)

        # Calc how many shots are needed to obtain an error within the given tolerance, given the estimated variance
        # from the first, intitial sample:
        shots = int(self.n_amp_factors * (gamma_coeff**2)*variances[0] / (error_tol**2))

        if verbose:
            print("Initial shots performed:", sample_shots,"| Estimated variance, sigma^2 = {:.4f}".format(variances[0]))
            print("To obtain an error within s =", error_tol,"need a total of", shots,"shots.")

        if self.error_detect:
            shots = int(shots / (1 - discarded_rates[0]))
            if verbose:
                print("Rate of discarded experiments = {:.4f}".format(discarded_rates[0]), "| new shot count =", shots)

        if shots > max_shots:
            if verbose:
                print("Exceeded maximal shot count, max_shots = {:}".format(str(max_shots)))
            shots = max_shots

        if shots <= sample_shots:
            if verbose:
                print("Already performed enough shots")
            return energies[0], variances[0], discarded_rates[0], shot_samples[0]

        shot_samples[1] = shots - shot_samples[0]

        energies[1], variances[1], discarded_rates[1] = self.compute_energy_sample(noise_amplified_ansatz, shot_samples[1])

        # Weighted average over both samples
        noise_amplified_energy = (shot_samples[0]/shots) * energies[0] + (shot_samples[1]/shots) * energies[1]
        variance = (shot_samples[0]/shots) * variances[0] + (shot_samples[1]/shots) * variances[1]
        discarded_rate = (shot_samples[0]/shots) * discarded_rates[0] + (shot_samples[1]/shots) * discarded_rates[1]

        return noise_amplified_energy, variance, discarded_rate, shots

    # COMPLETE VQE FUNCTIONS

    def objective_function(self, params: Union[np.ndarray, list], verbose: bool = None,
                           save_noise_amplified_results: bool = False):
        if verbose is None:
            verbose = self.verbose

        # Specify the parameter values in the parameterized ansatz
        ansatz = self.ansatz.bind_parameters({self.parameters: params})

        self.noise_amplified_energies = np.zeros(self.n_amp_factors)
        self.discarded_rates = np.zeros(self.n_amp_factors)
        self.variances = np.zeros(self.n_amp_factors)
        self.performed_shots = np.zeros(self.n_amp_factors)
        amplification_factors = np.asarray([2*i + 1 for i in range(self.n_amp_factors)])

        if verbose:
            print("____\nn_amp_factors={:}, error_detection={:}".format(str(self.n_amp_factors), str(self.error_detect)))

        for i, amp_factor in enumerate(amplification_factors):
            if verbose:
                print("executing circuits, amp_factor={:}".format(amp_factor))

            gamma_coeff = self.gamma_coeffs[i]

            results_read = False
            if save_noise_amplified_results:
                noise_amplified_energy, variance, discarded_rate, performed_shots = \
                    self.read_noiseamplified_result(params, backend_name=self.backend.name(), shots=self.shots,
                                                    error_controlled_sampling=self.error_controlled_sampling,
                                                    error_tol=self.error_tol, max_shots=self.max_shots,
                                                    n_amp_factors=self.n_amp_factors, amp_factor=amp_factor,
                                                    directory=self.directory)
                if not ((noise_amplified_energy is None) or (variance is None)
                        or (discarded_rate is None) or (performed_shots is None)):
                    results_read = True
                    if verbose:
                        print("Noise amplified results for r={:} read from file".format(amp_factor))
                elif verbose:
                    print("Noise amplified results for r={:} not found. Computing".format(amp_factor))

            if not results_read:
                noise_amplified_energy, variance, discarded_rate, performed_shots = \
                    self.compute_noise_amplified_energy(ansatz, error_tol=self.error_tol, gamma_coeff=gamma_coeff,
                                                        amp_factor=amp_factor, sample_shots=self.shots,
                                                        max_shots=self.max_shots, verbose=verbose)
                if save_noise_amplified_results:
                    self.write_noiseamplified_result(noise_amplified_energy, variance, discarded_rate, performed_shots,
                                                     params, backend_name=self.backend.name(), shots=self.shots,
                                                     error_controlled_sampling=self.error_controlled_sampling,
                                                     error_tol=self.error_tol, max_shots=self.max_shots,
                                                     n_amp_factors=self.n_amp_factors, amp_factor=amp_factor,
                                                     directory=self.directory)
                    if verbose:
                        print("Noise amplified results for r={:} successfully written to file".format(amp_factor))

            self.noise_amplified_energies[i] = noise_amplified_energy
            self.variances[i] = variance
            self.discarded_rates[i] = discarded_rate
            self.performed_shots[i] = performed_shots

        self.iter += 1

        if verbose:
            print("E_r =", self.noise_amplified_energies, "___ params =", params, "___ iter =", self.iter)

        return self.richardson_extrapolate(self.noise_amplified_energies)

    def compute_energy(self, params: Union[np.ndarray, list], save_results: bool = None, verbose: bool = None):
        if save_results is None:
            save_results = self.save_results
        if verbose is None:
            verbose = self.verbose

        if save_results:

            energy, noise_amplified_energies, variances, discarded_rates, performed_shots =\
                self.read_energycomputation_result(params, self.backend.name(), self.shots, self.n_amp_factors,
                                                   self.error_controlled_sampling, self.error_tol, self.max_shots,
                                                   directory=self.directory)

            if not (energy is None):
                self.noise_amplified_energies = noise_amplified_energies
                self.variances = variances
                self.discarded_rates = discarded_rates
                self.performed_shots = performed_shots

                return energy

        if save_results:
            print("results could not be read from file, computing")

        if save_results and self.n_amp_factors > 1:
            print("Saving of partial, noise amplified results on")
            save_noise_amplified_results = True
        else:
            save_noise_amplified_results = False

        energy = self.objective_function(params=params, verbose=verbose,
                                         save_noise_amplified_results=save_noise_amplified_results)

        if save_results:
            self.write_energycomputation_result(energy, self.noise_amplified_energies, self.variances, self.discarded_rates,
                                                self.performed_shots, params, self.backend.name(), self.shots,
                                                self.n_amp_factors, self.error_controlled_sampling, self.error_tol,
                                                self.max_shots, directory=self.directory)

        return energy

    def compute_vqe(self, x0: np.ndarray, method: str = "COBYLA", tol=0.01, verbose: bool = False, bounds=None) -> (float, np.ndarray):

        self.verbose = verbose
        num_params = np.shape(x0)[0]

        self.iter = 0

        if bounds is None:
            bounds = [(0., 2.*np.pi) for i in range(num_params)]

        # If save_results==True, try to read results from file
        if self.save_results:
            energy, optimal_params = self.read_result(backend_name=self.backend.name(), shots=self.shots,
                                                      method=method, tol=tol, n_amp_factors=self.n_amp_factors,
                                                      error_controlled_sampling=self.error_controlled_sampling,
                                                      error_tol=self.error_tol, max_shots=self.max_shots)
            if not (energy is None or optimal_params is None):
                print("results read from file")
                self.energy, self.optimal_params = energy, optimal_params
                return energy, optimal_params

        if self.verbose:
            print("Starting VQE, method =",method,"_ tol =",tol,"_ x0 =",x0)

        # Perform the optimization procedure
        if method == "SPSA":
            opt = SPSA()
            params, energy, _ = opt.optimize(num_params, objective_function=self.objective_function,
                                             initial_point=x0, variable_bounds=bounds)
            self.energy, self.optimal_params = energy, params
        else:
            res = minimize(self.objective_function, x0=x0, method=method, tol=tol)
            self.energy, self.optimal_params = res.fun, res.x

        if self.save_results:
            self.write_result(self.energy, self.optimal_params, backend_name=self.backend.name(), shots=self.shots,
                              method=method, tol=tol, n_amp_factors=self.n_amp_factors,
                              error_controlled_sampling=self.error_controlled_sampling, error_tol=self.error_tol,
                              max_shots=self.max_shots)
        return self.energy, self.optimal_params

    # Complete VQE functions, using statevector simulations

    def objective_function_statevector(self, params: np.ndarray, backend, qubit_op, verbose):
        ansatz = self.ansatz.bind_parameters({self.parameters: params})
        job = execute(ansatz, backend)
        state = job.result().get_statevector()

        e, sigma = qubit_op.evaluate_with_statevector(state)

        energy = np.real(e) + self.nuclear_repulsion_energy

        if verbose:
            print(energy, "-", params)

        return energy

    def compute_statevector_vqe(self, x0: np.ndarray, method: str = "COBYLA", tol=0.01, verbose=False) -> (float, np.ndarray):
        statevector_backend = Aer.get_backend("statevector_simulator")

        # If save_results==True, try to read results from file
        if self.save_results:
            energy, optimal_params = self.read_result(backend_name="statevector_simulator", shots=1,
                                                      method=method, tol=tol, n_amp_factors=1,
                                                      error_controlled_sampling=False, error_tol=0, max_shots=1)
            if not (energy is None or optimal_params is None):
                print("results read from file")
                self.energy, self.optimal_params = energy, optimal_params
                return energy, optimal_params

        # Perform the optimization procedure
        res = minimize(self.objective_function_statevector, x0=x0, method=method, tol=tol,
                       args=(statevector_backend, self.qubit_operator, verbose))

        self.energy, self.optimal_params = res.fun, res.x
        if self.save_results:
            self.write_result(self.energy, self.optimal_params,backend_name="statevector_simulator", shots=1,
                              method=method, tol=tol, n_amp_factors=1,error_controlled_sampling=False, error_tol=0,
                              max_shots=1)
        if verbose:
            print("E_g =", self.energy)
            print("Optimal params =", self.optimal_params)

        return self.energy, self.optimal_params

    # Compute the exact energy

    def compute_exact_energy(self):
        result = NumPyEigensolver(self.qubit_operator).run()
        return np.real(result.eigenvalues) + self.nuclear_repulsion_energy

import matplotlib.pyplot as plt

if __name__ == "__main__":

    from qiskit.test.mock import FakeAthens
    mock_backend = FakeAthens()
    sim_backend = Aer.get_backend("qasm_simulator")

    optimal = np.asarray([4.94560539, 5.75887687, 4.73019389, 7.02571097, 7.84432941, 2.39695389,
                          6.28253994, 3.26181875, 4.71907635, 5.19797265])

    h2_geometry = [("H",[0,0,0]), ("H",[0,0,0.74])]

    from sym_preserving_state_ansatze import get_n4_m2_particlepreserving_ansatz, get_n4_m2_parameterized_ansatz

    ansatz, params = get_n4_m2_parameterized_ansatz()
    ansatz_statevec, params_statevec = get_n4_m2_parameterized_ansatz(n_qubits=4, n_cbits=4)

    def get_error_detect_qc():
        qc = QuantumCircuit(5,5)
        qc.cx(0,4)
        qc.cx(1,4)
        qc.cx(2,4)
        qc.cx(3,4)
        qc.measure(4,4)
        return qc

    error_detect_qc = get_error_detect_qc()

    N_AMP_FACTORS = 3
    vqe = VQE(h2_geometry, ansatz=ansatz.copy(), parameters=params, shots=5*8192, backend=mock_backend, n_amp_factors=N_AMP_FACTORS)
              #error_detect_qc=get_error_detect_qc(), error_detect_qubit=4)#,
              #error_detect_qc=get_error_detect_qc(), error_detect_qubit=4)
    #vqe_2 = VQE(h2_geometry, ansatz_2, shots=10*8192)

    vqe_2 = VQE(h2_geometry, ansatz=ansatz.copy(), parameters=params, shots=5 * 8192, backend=mock_backend, n_amp_factors=N_AMP_FACTORS,
                error_detect_qc=error_detect_qc.copy(), error_detect_qubits=[4])

    x0 = np.asarray([5.03313128, 5.32006811, 4.85125363, 5.58864516, 7.64440043, 2.92400644,
                     4.54379885, 4.30882066, 3.08813554, 3.84747231])

    #print(vqe.objective_function(params=optimal))
    #print(vqe.objective_function(x0))
    #print(vqe_2.objective_function(x0))

    #print(vqe.compute_vqe(x0=x0, method="COBYLA"))

    #res = vqe.compute_statevector_vqe(x0=x0)
    vqe_statevec = VQE(h2_geometry, ansatz=ansatz_statevec, parameters=params_statevec, backend=sim_backend)

    vqe_statevec.compute_statevector_vqe(x0=x0)

    print(vqe_statevec.energy, "-", vqe_statevec.optimal_params)

    print(vqe.objective_function(params=vqe_statevec.optimal_params))
    print(vqe_2.objective_function(params=vqe_statevec.optimal_params))

    #statevector_backend = Aer.get_backend("statevector_simulator")
    #res = vqe_statevec.objective_function_statevector(params=optimal[::-1], backend=statevector_backend, qubit_op=vqe_statevec.qubit_operator)
    #print(res)
    #res = vqe.objective_function(params=optimal)
    #print(res)

    #print(res.x)
    #print(res.fun)