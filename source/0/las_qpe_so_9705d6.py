# https://github.com/rdcunha/las-qpe/blob/f5a8ee41143333dff3fc9c8dcbd3cd9535f43cd0/data/c4h6/las-qpe_so.py
import numpy as np
import logging
from argparse import ArgumentParser
# PySCF imports
from pyscf import gto, scf, lib, mcscf, ao2mo
from pyscf.tools import fcidump
from mrh.my_pyscf.fci import csf_solver
# mrh imports
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf.mcscf.lasci import h1e_for_cas
from c4h6_struct import structure

# Qiskit imports
from qiskit_nature.properties.second_quantization.electronic import (
    ElectronicStructureDriverResult,
    ElectronicEnergy,
    ParticleNumber,
)
from qiskit_nature.properties.second_quantization.electronic.integrals import (
    OneBodyElectronicIntegrals,
    TwoBodyElectronicIntegrals,
)
from qiskit_nature.properties.second_quantization.electronic.bases import ElectronicBasis
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper, ParityMapper
from qiskit.providers.aer import StatevectorSimulator
from qiskit import Aer
from qiskit import QuantumCircuit
from qiskit.utils import QuantumInstance
from qiskit_nature.algorithms import VQEUCCFactory, GroundStateEigensolver
from qiskit_nature.circuit.library import HartreeFock
from qiskit.algorithms import NumPyEigensolver, PhaseEstimation, PhaseEstimationScale
from qiskit.opflow import PauliTrotterEvolution,SummedOp,PauliOp,MatrixOp,PauliSumOp,StateFn

parser = ArgumentParser(description='Do QPE with a LAS reference')
parser.add_argument('--an', type=int, default=1, help='number of ancilla qubits')
parser.add_argument('--shots', type=int, default=1024, help='number of shots for the simulator')
args = parser.parse_args()

# Define molecule: C_4H_6
norb = 8
nelec = 8
norb_f = (4,4)
nelec_f = ((2,2),(2,2))
#xyz = '''C -1.83337627e+00  1.52864213e-16  3.62184378e-01
#    H -2.78128420e+00  5.34283450e-16 -1.39872751e-01
#    H -1.85755235e+00 -3.14359819e-16  1.43664264e+00
#    C -6.67087511e-01  2.55377626e-16 -3.23583629e-01
#    H -6.80782100e-01  7.29913533e-16 -1.40003063e+00
#    C  6.67087511e-01 -2.55377626e-16  3.23583629e-01
#    H  6.80782100e-01 -7.29913533e-16  1.40003063e+00
#    C  1.83337627e+00 -1.52864213e-16 -3.62184378e-01
#    H  1.85755235e+00  3.14359819e-16 -1.43664264e+00
#    H  2.78128420e+00 -5.34283450e-16  1.39872751e-01'''
#mol = gto.M (atom = xyz, basis = '6-31g', output='c4h6_631g_{}_{}.log'.format(args.an, args.shots),
#    symmetry=False, verbose=lib.logger.DEBUG)
mol = structure (0.0, 0.0, output='c4h6_631g_{}_{}.log'.format(args.an, args.shots), verbose=lib.logger.DEBUG)

# Do RHF
mf = scf.RHF(mol).run()
print("HF energy: ", mf.e_tot)

#CASSCF (for comparison with Matt's code)
mc = mcscf.CASSCF (mf, norb, nelec).set (fcisolver = csf_solver (mol, smult=1))
mo_coeff = mc.sort_mo ([11,12,14,15,16,17,21,24])
mc.kernel (mo_coeff)
mo_coeff = mc.mo_coeff.copy ()

# Create LASSCF object
las = LASSCF (mf, norb_f, nelec_f, spin_sub=(1,1)).set (mo_coeff=mo_coeff)

# Localize the chosen fragment active spaces
loc_mo_coeff = las.localize_init_guess ([[0,1,2,3,4],[5,6,7,8,9]])
las.kernel(loc_mo_coeff)
loc_mo_coeff = las.mo_coeff
print("LASSCF energy: ", las.e_tot)
#loc_mo_coeff = mf.mo_coeff

ncore = las.ncore
ncas = las.ncas
ncas_sub = las.ncas_sub
nelec_cas = las.nelecas

print ("Ncore: ", ncore, "Ncas: ", ncas, "Ncas_sub: ", ncas_sub, "Nelec_cas: ", nelec_cas)

# Situation so far: we have mf.mo_coeffs containing [:,ncore:nsub1:nsub2:next]
# Creating a list of slices for core, subspace1, subspace2, etc
idx_list = [slice(0,ncore)]
prev_sub_size = 0
for i, sub in enumerate(ncas_sub):
    idx_list.append(slice(ncore+prev_sub_size, ncore+prev_sub_size + sub))
    prev_sub_size += sub

# To prepare: H_eff = \sum_K H_frag(K)
# H_frag(K) = h1'_k1^{k2} a^+_{k1} a_{k2} + 1/4 h2_{k2 k4}^{k1 k3} a^+_{k1} a^+_{k3} a_{k4} a_{k2}
# with h1' = h1_{k1}^{k2} + \sum_i h2_{k2 i}^{k1 i} + \sum{L \neq K} h2_{k2 l2}^{k1 l1} D_{l2}^{l1}

# First, construct D and ints
# Option 1: AO-basis HF 1-RDM to localized MO basis
'''
D = mf.make_rdm1(mo_coeff=mf.mo_coeff)
D_mo = np.einsum('pi,pq,qj->ij', loc_mo_coeff, D, loc_mo_coeff)
'''
# Option 2: Converged LAS 1-RDM
D_mo = las.make_rdm1(mo_coeff=las.mo_coeff)
# Convert to spin orbitals
D_so = np.block([[D_mo, np.zeros_like(D_mo)],[np.zeros_like(D_mo), D_mo]])
#print("D:\n",D_so)

hcore_ao = mf.get_hcore(mol)
hcore_mo = np.einsum('pi,pq,qj->ij', loc_mo_coeff, hcore_ao, loc_mo_coeff)
# Convert to spin orbitals
hcore_so = np.block([[hcore_mo, np.zeros_like(hcore_mo)],[np.zeros_like(hcore_mo), hcore_mo]])

nso = mol.nao_nr() * 2
eri_4fold = ao2mo.kernel(mol.intor('int2e'), mo_coeffs=loc_mo_coeff)
eri = ao2mo.restore(1, eri_4fold,mol.nao_nr())
#print(eri[0])
# Convert to spin orbitals
eri_so = np.zeros((nso, nso, nso, nso))
one_indices = (
            (0, 0, 0, 0),  # alpha-alpha-spin
            (0, 1, 1, 0),  # beta-alpha-spin
            (1, 1, 1, 1),  # beta-beta-spin
            (1, 0, 0, 1),  # alpha-beta-spin
        )
for one_idx in one_indices: 
    ao_mat = np.einsum('ijkl->ljik', eri)  # physicist notation
    kron = np.zeros((2,2,2,2))
    kron[one_idx] = 1
    eri_so -= 0.5 * np.kron(kron, ao_mat)
#print(eri_so[0])

# Storing each fragment's h1 and h2 as a list
h1_frag = []
h2_frag = []

# Then construct h1' for each fragment
# and for alpha-alpha, beta-beta blocks
for idx in idx_list[1:]:
    print(idx_list[0])
    inactive_mask = idx_list[0]
    eri_mix = eri_so[:,inactive_mask,:,inactive_mask].copy()
    print(eri_mix.shape)
    h1p = hcore_so[idx,idx].copy()
    if h1p.size == 0:
        h1p = np.einsum('jiki->jk', eri_mix[idx,:,idx,:])
        for i,idx2 in enumerate(idx_list):
            if i > 0 and idx2 != idx:
                h1p += np.einsum('ikjl,kl->ij', eri_so[idx,idx2,idx,idx2],D_so[idx2,idx2])
    else:
        h1p += np.einsum('jiki->jk', eri_mix[idx,:,idx,:])
        for i,idx2 in enumerate(idx_list):
            if i > 0 and idx2 != idx:
                h1p += np.einsum('ikjl,kl->ij', eri_so[idx,idx2,idx,idx2],D_so[idx2,idx2])

    # Finally, construct total H_frag
    h1_frag.append(h1p)
    h2_frag.append(eri[idx,idx,idx,idx])

# using the built-in LASCI function h1e_for_cas
h1_frag_2 = h1e_for_cas(las)

# Just using h1e_for_cas as my fragment h1
h1_frag = []
for f in range(len(ncas_sub)):
    h1_frag.append(h1_frag_2[f][0][0])
print("h1e: {}".format(h1_frag))

for f in range(len(ncas_sub)):
    print("H1_frag shape: ", h1_frag[f].shape)
    print("H2_frag shape: ", h2_frag[f].shape)

# Function below stolen from qiskit's Hamiltonian Phase Estimation class
def _remove_identity(pauli_sum):
    """Remove any identity operators from `pauli_sum`. Return
    the sum of the coefficients of the identities and the new operator.
    """
    idcoeff = 0.0
    ops = []
    for op in pauli_sum:
        p = op.primitive
        if p.x.any() or p.z.any():
            ops.append(op)
        else:
            idcoeff += op.coeff

    return idcoeff, SummedOp(ops)

phases_list = []
en_list = []
total_op_list = []

for frag in range(len(ncas_sub)):
    # WARNING: these have to be set manually for each fragment!
    num_alpha = int(nelec_cas[0] / 2)
    num_beta = int(nelec_cas[1] / 2)

    # For QPE, need second_q_ops
    # Hacking together an ElectronicStructureDriverResult to create second_q_ops
    # Lines below stolen from qiskit's FCIDump driver and modified
    particle_number = ParticleNumber(
        num_spin_orbitals=ncas_sub[frag]*2,
        num_particles=(num_alpha, num_beta),
    )

    # Assuming an RHF reference for now, so h1_b, h2_ab, h2_bb are created using 
    # the corresponding spots from h1_frag and just the aa term from h2_frag
    print("Nuclear repulsion: ", las.energy_nuc())
    electronic_energy = ElectronicEnergy(
        [
            # Using MO basis here for simplified conversion
            OneBodyElectronicIntegrals(ElectronicBasis.MO, (h1_frag[frag], None)),
            TwoBodyElectronicIntegrals(ElectronicBasis.MO, (h2_frag[frag], h2_frag[frag], h2_frag[frag], None)),
        ],
        nuclear_repulsion_energy=las.energy_nuc(),
    )

    # QK NOTE: under Python 3.6, pylint appears to be unable to properly identify this case of
    # nested abstract classes (cf. https://github.com/Qiskit/qiskit-nature/runs/3245395353).
    # However, since the tests pass I am adding an exception for this specific case.
    # pylint: disable=abstract-class-instantiated
    driver_result = ElectronicStructureDriverResult()
    driver_result.add_property(electronic_energy)
    driver_result.add_property(particle_number)

    second_q_ops = driver_result.second_q_ops()
    second_q_ops[0].set_truncation(0)
    #print(second_q_ops[0])

    # Choose fermion-to-qubit mapping
    qubit_converter = QubitConverter(mapper = ParityMapper(), two_qubit_reduction=True)
    # This just outputs a qubit op corresponding to a 2nd quantized op
    qubit_ops = [qubit_converter.convert(op) for op in second_q_ops]
    hamiltonian = qubit_ops[0]
    #print(hamiltonian)

    # Set the backend
    quantum_instance = QuantumInstance(backend = Aer.get_backend('aer_simulator'), shots=args.shots, optimization_level=0)

    np_solver = NumPyEigensolver(k=1)
    ed_result = np_solver.compute_eigenvalues(hamiltonian)
    print("NumPy result: ", ed_result.eigenvalues)

    # Can choose a regular solver from qiskit.algorithms
    qpe_solver = PhaseEstimation(num_evaluation_qubits=args.an, quantum_instance=quantum_instance)

    # Create a unitary out of the Hamiltonian ops
    # Lines below stolen from qiskit.algorithms.HPE
    if isinstance(hamiltonian, PauliSumOp):
        hamiltonian = hamiltonian.to_pauli_op()
    elif isinstance(hamiltonian, PauliOp):
        hamiltonian = SummedOp([hamiltonian])

    if isinstance(hamiltonian, SummedOp):
        id_coefficient, hamiltonian_no_id = _remove_identity(hamiltonian)
    else:
        raise TypeError("Hamiltonian must be PauliSumOp, PauliOp or SummedOp.")

    pe_scale = PhaseEstimationScale.from_pauli_sum(hamiltonian_no_id)
    print("Bound: ",pe_scale._bound)

    # QK: scale so that phase does not wrap.
    scaled_hamiltonian = -pe_scale.scale * hamiltonian_no_id  

    # Default evolution: PauliTrotterEvolution
    evolution = PauliTrotterEvolution()

    # Create the unitary by evolving the Hamiltonian
    unitary = evolution.convert(scaled_hamiltonian.exp_i())

    if not isinstance(unitary, QuantumCircuit):
        unitary_circuit = unitary.to_circuit()
    else:
        unitary_circuit = unitary

    # QK: Decomposing twice allows some 1Q Hamiltonians to give correct results
    # QK: when using MatrixEvolution(), that otherwise would give incorrect results.
    # QK: It does not break any others that we tested.
    unitary = unitary_circuit.decompose().decompose()

    # Printing this is not a good idea because the circuit is very large
    #print(unitary)

    # Create an HF initial state and add it to the estimate function
    # For our (H2)_2 system 8 spin orbs, 2 alpha 2 beta electrons
    init_state = HartreeFock(ncas_sub[frag]*2, (num_alpha,num_beta), qubit_converter)

    circuit = qpe_solver.construct_circuit(unitary=unitary, state_preparation=init_state).decompose().decompose()

    op_dict = circuit.count_ops()
    total_ops = sum(op_dict.values())
    total_op_list.append(total_ops)
    print("Operations: {}".format(op_dict))
    print("Total operations: {}".format(total_ops))
    print("Nonlocal gates: {}".format(circuit.num_nonlocal_gates()))

    # Estimate takes in a SummedPauli or a PauliOp and outputs a scaled estimate of the eigenvalue
    res = qpe_solver.estimate(unitary=unitary, state_preparation=init_state)
    phases = res.__dict__['_phases']
    phases_list.append(phases)
    print(res)
    scaled_phases = pe_scale.scale_phases(res.filter_phases(cutoff=0.0, as_float=True), id_coefficient=id_coefficient)
    scaled_phases = {v:k for k, v in scaled_phases.items()}
    print(scaled_phases)
    energy_dict = {k:scaled_phases[v] for k, v in phases.items()}
    en_list.append(energy_dict)
    most_likely_eig = scaled_phases[max(scaled_phases.keys())]
    most_likely_an = max(phases, key=phases.get)
    print("Most likely eigenvalue: ", most_likely_eig)
    print("Most likely ancilla sign: ", most_likely_an)

np.save('results_{}_{}.npy'.format(args.an, args.shots), {'n_frag':len(ncas_sub), 'phases':phases_list, 'energies':en_list, 'operations':total_op_list})
print("Phases: ",phases_list)
print("en_list: ",en_list)
print("total_op_list: ",total_op_list)

