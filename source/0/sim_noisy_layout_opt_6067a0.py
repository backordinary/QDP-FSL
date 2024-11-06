# https://github.com/beneklins/qiskiteers_h2/blob/ed989ae074f19f8ff5153d92bc2eed3453c80fad/scripts/sim_noisy_layout_opt.py
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

# create variational circuit
varform_4qubits_1param = QuantumCircuit(4)
a = Parameter('a')
varform_4qubits_1param.ry(a, 1)
varform_4qubits_1param.x(0)
varform_4qubits_1param.cx(1, 0)
varform_4qubits_1param.cx(0, 2)
varform_4qubits_1param.cx(1, 3)

# H2 at ~ the equilibrium bond length (0.735 Angstrom)
with open('Integrals_sto-3g_H2_d_0.7350_no_spin.npz', 'rb') as f:
    out = np.load(f)
    h1_load_no_spin = out['h1']
    h2_load_no_spin = out['h2']
molecular_hamiltonian = MolecularFermionicHamiltonian.from_integrals(h1_load_no_spin, h2_load_no_spin).include_spin()

# map to the quantum computer
mapping = JordanWigner()
lcps_h2 = mapping.fermionic_hamiltonian_to_linear_combinaison_pauli_string(
    molecular_hamiltonian).combine().apply_threshold().sort()

# Run the Simulator with no noise a bunch of times
qasm_simulator = Aer.get_backend('qasm_simulator')

# Minimizer for the Variational Circuit parameter
minimizer = lambda fct, start_param_values: minimize(
    fct,
    start_param_values,
    method='SLSQP',
    options={'maxiter': 5, 'eps': 1e-1, 'ftol': 1e-4})


def get_energies(N, shots):
    execute_opts = {'shots': shots,
                    'noise_model': bogota_nm,
                    'coupling_map': bogota_conf.coupling_map,
                    'basis_gates': bogota_conf.basis_gates,
                    'initial_layout': [2, 3, 1, 4]}
    evaluator = BasicEvaluator(varform_4qubits_1param, qasm_simulator, execute_opts=execute_opts, measure_filter=None)
    vqe_solver = VQESolver(evaluator, minimizer, [0, ], name='vqe_solver')

    electronic_energies = np.zeros(N)
    for i in range(N):
        electronic_energies[i], _ = vqe_solver.lowest_eig_value(lcps_h2)
        print(
            f"Mean electronic energy after {i + 1} simulations: {electronic_energies[:i + 1].sum() / (i + 1)}        ",
            end='\r')

    print("Done!                                                                                                   \n ")
    #
    # print("Electronic Energies:")
    # print(electronic_energies)
    # print(f"Mean Electronic Energy Calculated from Noisy Simulator, 1024 shots, {N} Times: {electronic_energies.mean()} Eh")
    # print(f"Standard Deviation : {electronic_energies.std()} Eh")
    return electronic_energies

if __name__ == "__main__":
    N = 50
    exact_e = -1.8572750302023788
    for shots in [100, 500, 1024, 4096, 8192]:
        print(f'Running with {shots} shots...')
        electronic_energies = get_energies(N, shots)
        with open(f'sim_noisy_layout_opt_{shots}_shots.npz', 'wb') as f:
            np.savez(f, distance=0.735,
                     electronic_energies=electronic_energies,
                     exact_electronic_energy=exact_e,
                     shots=shots,
                     number_of_sims=N,
                     mean_energy=electronic_energies.mean(),
                     std_energy=electronic_energies.std(),
                     error=np.abs((electronic_energies.mean() - exact_e) / exact_e))