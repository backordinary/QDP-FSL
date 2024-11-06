# https://github.com/beneklins/qiskiteers_h2/blob/ed989ae074f19f8ff5153d92bc2eed3453c80fad/scripts/dissociation_curve.py
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import Parameter
from evaluator import Evaluator
from pauli_string import PauliString
from evaluator import BasicEvaluator
from qiskit import Aer, execute
from hamiltonian import MolecularFermionicHamiltonian
from mapping import JordanWigner
from scipy.optimize import minimize
from solver import VQESolver
from pyscf import gto
from qiskit.providers.aer.noise import NoiseModel
from qiskit.circuit import QuantumRegister
from qiskit.ignis.mitigation.measurement import complete_meas_cal
from qiskit.ignis.mitigation.measurement import CompleteMeasFitter
from qiskit import IBMQ

# Load IBM Account and Provider
IBMQ.enable_account(
    '756e275ccf50985fef8b93c4b4732c4c2ec4f8b491fe142093e9a3bb8d8f44044727d2ac28ac87ea96d4e9d95e7884c5fcdfe4f00942c236286f419c1145f149')
provider = IBMQ.get_provider(hub='ibm-q-education', group='qscitech-quantum', project='qc-bc-workshop')

# Bogota info
bogota = provider.get_backend('ibmq_bogota')
bogota_prop = bogota.properties()
bogota_conf = bogota.configuration()
bogota_nm = NoiseModel.from_backend(bogota_prop)

# Run the Simulator
qasm_simulator = Aer.get_backend('qasm_simulator')

# Measurement Calibration
qr = QuantumRegister(4)
qubit_list = [0, 1, 2, 3]
meas_calibs, state_labels = complete_meas_cal(qubit_list=qubit_list, qr=qr, circlabel='mcal')
calibration_layout = [2, 3, 1, 4]
result = execute(meas_calibs,
                 qasm_simulator,
                 shots=8192,
                 noise_model=bogota_nm,
                 coupling_map=bogota_conf.coupling_map,
                 basis_gates=bogota_conf.basis_gates,
                 initial_layout=calibration_layout).result()
meas_fitter = CompleteMeasFitter(result, state_labels)
meas_filter = meas_fitter.filter

# create variational circuit
varform_4qubits_1param = QuantumCircuit(4)
a = Parameter('a')
varform_4qubits_1param.ry(a, 1)
varform_4qubits_1param.x(0)
varform_4qubits_1param.cx(1, 0)
varform_4qubits_1param.cx(0, 2)
varform_4qubits_1param.cx(1, 3)

# Minimizer for the Variational Circuit parameter
minimizer = lambda fct, start_param_values: minimize(
    fct,
    start_param_values,
    method='SLSQP',
    options={'maxiter': 5, 'eps': 1e-1, 'ftol': 1e-4})

def get_energies(N, shots, distance):
    # build the molecule and basis functions at set distance
    mol = gto.M(
        atom=[['H', (0, 0, -distance / 2)], ['H', (0, 0, distance / 2)]],
        basis='sto-3g'
    )
    # build the molecular Hamiltonian
    molecular_hamiltonian = MolecularFermionicHamiltonian.from_pyscf_mol(mol).include_spin()
    energy_nuc = mol.energy_nuc()

    # map to the quantum computer
    mapping = JordanWigner()
    lcps_h2 = mapping.fermionic_hamiltonian_to_linear_combinaison_pauli_string(
        molecular_hamiltonian).combine().apply_threshold().sort()

    execute_opts = {'shots': shots,
                    'noise_model': bogota_nm,
                    'coupling_map': bogota_conf.coupling_map,
                    'basis_gates': bogota_conf.basis_gates,
                    'initial_layout': [2, 3, 1, 4]}
    evaluator = BasicEvaluator(varform_4qubits_1param, qasm_simulator, execute_opts=execute_opts,
                               measure_filter=meas_filter)
    vqe_solver = VQESolver(evaluator, minimizer, [0, ], name='vqe_solver')

    electronic_energies = np.zeros(N)
    total_energies = np.zeros(N)
    for i in range(N):
        electronic_energies[i], _ = vqe_solver.lowest_eig_value(lcps_h2)
        total_energies[i] = electronic_energies[i] + energy_nuc
        print(f"Calculation {i} at distance {distance:4.3f}", end='\r')

    return electronic_energies, total_energies



if __name__ == '__main__':
    shots = 1024
    N = 20
    Nd = 20
    # try a range of internuclear distances
    distances = np.linspace(0.3, 2.5, Nd)
    total_energies = np.zeros((Nd, N))
    electronic_energies = np.zeros((Nd, N))

    for i, distance in enumerate(distances):  # units in AA
        print(f'Trying Distance {i+1} / {Nd}...')
        electronic_energies[i, :], total_energies[i, :] = get_energies(N, shots, distance)

    with open(f'h2_dissociation_noisy_sim_{shots}_shots.npz', 'wb') as f:
        np.savez(f, distances=distances,
                 electronic_energies=electronic_energies,
                 total_energies=total_energies,
                 shots=shots,
                 number_of_sims=N)
