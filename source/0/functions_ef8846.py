# https://github.com/jason-jk-kang/QIS-qchem/blob/6f296a4272d842a136892e33f8edd77d2051fc60/ProjectQ/functions.py
import os
import numpy
import itertools
from numpy import array, concatenate, zeros
from numpy.random import randn
from qiskit import *
from scipy.optimize import minimize
from openfermion.config import *
from openfermionprojectq import *
from openfermionprojectq._unitary_cc  import uccsd_evolution

from openfermion.hamiltonians import MolecularData
from openfermion.transforms import jordan_wigner, get_fermion_operator, get_sparse_operator
from openfermion.utils import uccsd_singlet_paramsize, uccsd_generator, uccsd_singlet_generator, uccsd_singlet_get_packed_amplitudes
from openfermion.ops import FermionOperator, QubitOperator, down_index, up_index

from projectq.ops import X, All, Measure
from projectq.backends import CommandPrinter, CircuitDrawer, IBMBackend, Simulator

from openqasm import OpenQASMEngine
from pyscf import mp, fci
import matplotlib.pyplot as plt
from openfermionpyscf import run_pyscf
from projectq.meta import insert_engine


def energy_objective(packed_amplitudes, molecule, qubit_hamiltonian, compiler_engine):
    """Evaluate the energy of a UCCSD singlet wavefunction with packed_amplitudes
    Args:
        packed_amplitudes(ndarray): Compact array that stores the unique
            amplitudes for a UCCSD singlet wavefunction.
    Returns:
        energy(float): Energy corresponding to the given amplitudes
    """
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    # Set Jordan-Wigner initial state with correct number of electrons, hartree fock state
    wavefunction = compiler_engine.allocate_qureg(molecule.n_qubits)
    for i in range(molecule.n_electrons):
        X | wavefunction[i]

    # Prepares the wavefunction on our packed_amplitudes, generates a circuit
    evolution_operator = uccsd_singlet_evolution(packed_amplitudes,
                                                 molecule.n_qubits,
                                                 molecule.n_electrons)
    evolution_operator | wavefunction
    compiler_engine.flush()

    # Evaluate the energy and reset wavefunction
    energy = compiler_engine.backend.get_expectation_value(qubit_hamiltonian, wavefunction)
    All(Measure) | wavefunction
    compiler_engine.flush()
    return energy


def run_simulation (system, indx, commandprinter = False):
    # Load saved file for H3.
    basis = 'sto-3g'
    spin = 1

    # Set Hamiltonian parameters.
    active_space_start = 1
    active_space_stop = 3

    # Set calculation parameters.
    run_scf = 1
    run_mp2 = 1
    run_cisd = 0
    run_ccsd = 1
    run_fci = 1
    delete_input = True
    delete_output = True

    # Begin Running Simulation, Convert distance_counter to angstroms
    if indx == None:
        geometry = [('H', (0., 0., system.atoms[0].position[-1] * 0.529177249)),
                    ('H', (0., 0., system.atoms[1].position[-1] * 0.529177249)),
                    ('H', (0., 0., system.atoms[2].position[-1] * 0.529177249))]
    elif indx == 0:
        geometry = [('H', (0., 0., system.atoms[0].stand_by_position * 0.529177249)),
                    ('H', (0., 0., system.atoms[1].position[-1] * 0.529177249)),
                    ('H', (0., 0., system.atoms[2].position[-1] * 0.529177249))]

    elif indx == 1:
        geometry = [('H', (0., 0., system.atoms[0].position[-1] * 0.529177249)),
                    ('H', (0., 0., system.atoms[1].stand_by_position * 0.529177249)),
                    ('H', (0., 0., system.atoms[2].position[-1] * 0.529177249))]

    elif indx == 2:
        geometry = [('H', (0., 0., system.atoms[0].position[-1] * 0.529177249)),
                    ('H', (0., 0., system.atoms[1].position[-1] * 0.529177249)),
                    ('H', (0., 0., system.atoms[2].stand_by_position * 0.529177249))]

    # Generate and populate instance of MolecularData.
    molecule = MolecularData(geometry, basis, spin, charge = 1, description="h3")

    molecule = run_pyscf(molecule,
                         run_scf=run_scf,
                         run_mp2=run_mp2,
                         run_cisd=run_cisd,
                         run_ccsd=run_ccsd,
                         run_fci=run_fci)

    # Use a Jordan-Wigner encoding, and compress to remove 0 imaginary components
    molecular_hamiltonian = molecule.get_molecular_hamiltonian()

    fermion_hamiltonian = get_fermion_operator(molecular_hamiltonian)
    qubit_hamiltonian = jordan_wigner(fermion_hamiltonian)
    qubit_hamiltonian.compress()
    compiler_engine = uccsd_trotter_engine()

    ' Here we choose how we initialize our amplitudes '
    # packed_amplitudes = uccsd_singlet_get_packed_amplitudes(
    #        molecule.ccsd_single_amps,
    #        molecule.ccsd_double_amps,
    #        molecule.n_qubits,
    #        molecule.n_electrons)
    #
    # # Initialize the VQE with UCCSD amplitudes
    # reference_state = array(packed_amplitudes)*(-1.)
    # print ('Reference State UCCSD amplitudes:', reference_state)

    # Initialize the VQE with previous optimal amplitudes
    reference_state = system.opt_amplitudes
    print(f"Reference State previous optima: {system.opt_amplitudes}")

    ' # # # # # # # # # # # # # # # # # # # # # # # # # # # '

    # Run VQE Optimization to find new CCSD parameters
    opt_result = minimize(energy_objective, reference_state, (molecule, qubit_hamiltonian, compiler_engine), method="CG", options={'disp':True})
    opt_energy, system.opt_amplitudes = opt_result.fun, opt_result.x
    print (f'VQE Final amplitude: {system.opt_amplitudes}')
    print (f'VQE energy: {opt_energy}')
    print (f'FCI energy: {molecule.fci_energy}')
    print (f'CCSD energy: {molecule.ccsd_energy}')
    print (f'SCF (Hartree-Fock) energy: {molecule.hf_energy}')

    if commandprinter:
        with open('commands.txt', 'a') as f:
            backend = OpenQASMEngine()
            # Print commands. But this circuit is only up to the point where you prepare the wavefunction for the optimized amplitudes
            compiler_engine = uccsd_trotter_engine(backend)
            wavefunction = compiler_engine.allocate_qureg(molecule.n_qubits)
            for i in range(molecule.n_electrons):
                X | wavefunction[i]
            evolution_operator = uccsd_singlet_evolution(system.opt_amplitudes,
                                                         molecule.n_qubits,
                                                         molecule.n_electrons)
            evolution_operator | wavefunction
            compiler_engine.flush()
            print(backend.circuit.qasm())

    return ({"Name" : molecule.name, "VQE Energy" : opt_energy,
             "FCI Energy" : molecule.fci_energy, "CCSD Energy": molecule.ccsd_energy})






def sep_uccsd_singlet_generator(packed_amplitudes, n_qubits, n_electrons,
                            anti_hermitian=True):
    """Create a singlet UCCSD generator for a system with n_electrons
    This function generates a FermionOperator for a UCCSD generator designed
        to act on a single reference state consisting of n_qubits spin orbitals
        and n_electrons electrons, that is a spin singlet operator, meaning it
        conserves spin.
    Args:
        packed_amplitudes(list): List storing the unique single
            and double excitation amplitudes for a singlet UCCSD operator.
            The ordering lists unique single excitations before double
            excitations.
        n_qubits(int): Number of spin-orbitals used to represent the system,
            which also corresponds to number of qubits in a non-compact map.
        n_electrons(int): Number of electrons in the physical system.
        anti_hermitian(Bool): Flag to generate only normal CCSD operator
            rather than unitary variant, primarily for testing
    Returns:
        generator(FermionOperator): Generator of the UCCSD operator that
            builds the UCCSD wavefunction.
    """
    if n_qubits % 2 != 0:
        raise ValueError('The total number of spin-orbitals should be even.')

    n_spatial_orbitals = n_qubits // 2
    n_occupied = int(numpy.ceil(n_electrons / 2))
    n_virtual = n_spatial_orbitals - n_occupied

    # Unpack amplitudes
    n_single_amplitudes = n_occupied * n_virtual
    # Single amplitudes
    t1 = packed_amplitudes[:n_single_amplitudes]
    # Double amplitudes associated with one spatial occupied-virtual pair
    t2_1 = packed_amplitudes[n_single_amplitudes:2 * n_single_amplitudes]
    # Double amplitudes associated with two spatial occupied-virtual pairs
    t2_2 = packed_amplitudes[2 * n_single_amplitudes:]

    # Initialize operator
    generator_single = FermionOperator()
    generator_double = FermionOperator()

    # Generate excitations
    spin_index_functions = [up_index, down_index]
    # Generate all spin-conserving single and double excitations derived
    # from one spatial occupied-virtual pair
    for i, (p, q) in enumerate(
            itertools.product(range(n_virtual), range(n_occupied))):

        # Get indices of spatial orbitals
        virtual_spatial = n_occupied + p
        occupied_spatial = q

        for spin in range(2):
            # Get the functions which map a spatial orbital index to a
            # spin orbital index
            this_index = spin_index_functions[spin]
            other_index = spin_index_functions[1 - spin]

            # Get indices of spin orbitals
            virtual_this = this_index(virtual_spatial)
            virtual_other = other_index(virtual_spatial)
            occupied_this = this_index(occupied_spatial)
            occupied_other = other_index(occupied_spatial)

            # Generate single excitations
            coeff = t1[i]
            generator_single += FermionOperator((
                (virtual_this, 1),
                (occupied_this, 0)),
                coeff)
            if anti_hermitian:
                generator_single += FermionOperator((
                    (occupied_this, 1),
                    (virtual_this, 0)),
                    -coeff)

            # Generate double excitation
            coeff = t2_1[i]
            generator_double += FermionOperator((
                (virtual_this, 1),
                (occupied_this, 0),
                (virtual_other, 1),
                (occupied_other, 0)),
                coeff)
            if anti_hermitian:
                generator_double += FermionOperator((
                    (occupied_other, 1),
                    (virtual_other, 0),
                    (occupied_this, 1),
                    (virtual_this, 0)),
                    -coeff)



    # Generate all spin-conserving double excitations derived
    # from two spatial occupied-virtual pairs
    for i, ((p, q), (r, s)) in enumerate(
            itertools.combinations(
                itertools.product(range(n_virtual), range(n_occupied)),
                2)):

        # Get indices of spatial orbitals
        virtual_spatial_1 = n_occupied + p
        occupied_spatial_1 = q
        virtual_spatial_2 = n_occupied + r
        occupied_spatial_2 = s

        # Generate double excitations
        coeff = t2_2[i]
        for (spin_a, spin_b) in itertools.product(range(2), repeat=2):
            # Get the functions which map a spatial orbital index to a
            # spin orbital index
            index_a = spin_index_functions[spin_a]
            index_b = spin_index_functions[spin_b]

            # Get indices of spin orbitals
            virtual_1_a = index_a(virtual_spatial_1)
            occupied_1_a = index_a(occupied_spatial_1)
            virtual_2_b = index_b(virtual_spatial_2)
            occupied_2_b = index_b(occupied_spatial_2)

            if virtual_1_a == virtual_2_b:
                continue
            if occupied_1_a == occupied_2_b:
                continue
            else:

                generator_double += FermionOperator((
                    (virtual_1_a, 1),
                    (occupied_1_a, 0),
                    (virtual_2_b, 1),
                    (occupied_2_b, 0)),
                    coeff)
                if anti_hermitian:
                    generator_double += FermionOperator((
                        (occupied_2_b, 1),
                        (virtual_2_b, 0),
                        (occupied_1_a, 1),
                        (virtual_1_a, 0)),
                        -coeff)

    return (generator_single, generator_double)



def sep_uccsd_singlet_evolution(packed_amplitudes, n_qubits, n_electrons,
                            fermion_transform=jordan_wigner):
    """Create a ProjectQ evolution operator for a UCCSD singlet circuit
    Args:
        packed_amplitudes(ndarray): Compact array storing the unique single
            and double excitation amplitudes for a singlet UCCSD operator.
            The ordering lists unique single excitations before double
            excitations.
        n_qubits(int): Number of spin-orbitals used to represent the system,
            which also corresponds to number of qubits in a non-compact map.
        n_electrons(int): Number of electrons in the physical system
        fermion_transform(openfermion.transform): The transformation that
            defines the mapping from Fermions to QubitOperator.
    Returns:
        evoution_operator(TimeEvolution): The unitary operator
            that constructs the UCCSD singlet state.
    """
    # Build UCCSD generator

    fermion_generator = sep_uccsd_singlet_generator(packed_amplitudes,
                                                    n_qubits,
                                                    n_electrons)

    evol = lambda x: uccsd_evolution(fermion_generator[x], fermion_transform)


    return {"singles" : evol(0),
            "doubles": evol(1)}
