# https://github.com/amarjahin/kitaev_models_vqe/blob/839a1231c8870ab3f4c2015ec3ba6210fc4d0895/fgvqe.py
from numpy import zeros,zeros_like, array, conjugate, pi, savetxt, linspace
from numpy.linalg import eigh
from scipy.optimize import minimize
from scipy.sparse.linalg import eigsh 
from qiskit import QuantumCircuit, transpile
from qiskit.providers.aer import QasmSimulator, StatevectorSimulator
from qiskit.algorithms import NumPyEigensolver,NumPyMinimumEigensolver
from qiskit.circuit.library.standard_gates.rz import RZGate

from kitaev_models import KitaevModel
from qiskit_conversion import convert_to_qiskit_PauliSumOp
from ansatz import GBSU, PSU, PDU, PFDU, test_PDU
from cost import phys_energy_ev, energy_ev
from reduce_ansatz import reduce_params, reduce_ansatz
# from projector_op import projector

mes = NumPyMinimumEigensolver()
es = NumPyEigensolver(k=2)

L = (3,3)           # size of the lattice 
lattice_type = 'honeycomb_torus'
J = (1,1, 1) # pure Kitaev terms 

# lattice_type = 'square_octagon_torus'
# J = (1, 1, 2**(0.5)) # pure Kitaev terms 

k_array = linspace(0, 0.1, num=9)
exact_energy_array = zeros_like(k_array)
optimal_energy_array = zeros_like(k_array)
state_overlap_array = zeros_like(k_array)
nfev_array = zeros_like(k_array)
nit_array = zeros_like(k_array)

# edges=[(1,7), (11,13), (14, 0), (10,4)]
edges = [(0,5), (10,7), (12,17), (0,13), (4,17), (2,15)]

for i in range(len(k_array)):
    print('###############', k_array[i], '################')
    FH = KitaevModel(L=L, J=J,kappa_1=k_array[i],kappa_2=k_array[i], lattice_type=lattice_type) 

    m_u = FH.number_of_Dfermions_u
    m = FH.number_of_Dfermions
    active_qubits = [*range(m_u)]

    h = FH.two_fermion_hamiltonian(edges)
    h_u = FH.jw_hamiltonian_u(h) # the Jordan_Wigner transformed fermionic Hamiltonian

    qubit_op_u = convert_to_qiskit_PauliSumOp(h_u)
    hamiltonian_u = qubit_op_u.to_spmatrix()
    fermion_result = es.compute_eigenvalues(qubit_op_u)
    print(f'exact fermion energy: {fermion_result.eigenvalues[0].real}')
    energy_gap = (fermion_result.eigenvalues[1] - fermion_result.eigenvalues[0]).real
    print(f'energy gap: {energy_gap}')


    spin_ham = convert_to_qiskit_PauliSumOp(FH.spin_hamiltonian)
    spin_result = mes.compute_minimum_eigenvalue(spin_ham)
    print(f'exact spin ground energy: {spin_result.eigenvalue.real}')

    #######################################################################
    simulator = StatevectorSimulator()
    method = 'BFGS'

    qc = QuantumCircuit(m_u)
    qc.x([*range(m_u)])

    cost = lambda params: energy_ev(hamiltonian=hamiltonian_u ,simulator=simulator,
                    qc_c=qc,params=params).real
    print(f'the initial energy: {cost([])}')
    #########################################################################

    ansatz_terms_dict = {'a': GBSU(num_qubits=m_u, active_qubits=active_qubits, det=-1, steps=1,param_name='a'),
                        'b' : PFDU(num_qubits=m_u, fermion_qubits=active_qubits, steps=1, param_name='b'), 
                        }

    op_params = []
    nfev = 0
    nit = 0
    for key in ansatz_terms_dict:
        qc.append(ansatz_terms_dict[key].to_instruction(),qargs=active_qubits) # add next set of terms
        qc.barrier()

        print(f"num parameters: {qc.num_parameters}")
        qc = transpile(qc, simulator) 
        params0 = list(zeros(qc.num_parameters))
        params0[0:len(op_params)] = op_params
        print('optimizer is now running...')
        result = minimize(fun=cost, x0=params0,  method=method, tol=0.0001, options={'maxiter':None}) # run optimizer
        print(f"optimization success:{result['success']}")
        op_params = list(result['x']) # get optimal params 
        optimal_energy = result['fun']

        print(f'optimal energy: {optimal_energy}')
        nit = nit + result['nit']
        nfev = nfev + result['nfev']

    print(f"optimal - exact / gap: {(optimal_energy - fermion_result.eigenvalues[0].real) / energy_gap :.15f}")

    print('num of iterations: ', nit)
    print('num of evaluations: ', nfev)

    # savetxt('two_vortcies_init_params.txt', result['x'])


    if isinstance(simulator, StatevectorSimulator):
        op_qc = qc.bind_parameters(op_params)
        op_state = simulator.run(op_qc).result().get_statevector()
        # op_state = (projector_mat @ op_state) / (conjugate(op_state.T) @ projector_mat @ op_state)**(0.5)

        overlap_subspace = 0
        # for ip in range(len(fermion_result.eigenvalues)): 
        #     if round(optimal_energy, 5) == round(fermion_result.eigenvalues[ip].real, 5):
        prob = abs(conjugate(op_state.T) @ fermion_result.eigenstates[0].to_matrix())**2 
        overlap_subspace = overlap_subspace + prob
        # print(i)


        print(f"1 - |<exact|optimal>|^2 : {1 - overlap_subspace:.15f}")
        state_overlap_array[i] = 1 - overlap_subspace


    exact_energy_array[i] = fermion_result.eigenvalues[0].real
    optimal_energy_array[i] = optimal_energy
    nfev_array[i] = nfev
    nit_array[i] = nit

savetxt('./result/exact_energy.out', exact_energy_array)
savetxt('./result/optimal_energy.out', optimal_energy_array)
savetxt('./result/overlap.out', state_overlap_array)
savetxt('./result/nfev.out', nfev_array)
savetxt('./result/nit.out', nit_array)
