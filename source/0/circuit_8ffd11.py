# https://github.com/wmkirby1/CS-VQE/blob/9a0a7634dcb77f064957c772cf229b7103cce3a8/cs_vqe/circuit.py
import os
from numpy.lib.npyio import _save_dispatcher
from cs_vqe.model import cs_vqe_model
import utils.bit_tools as bit
import utils.cs_vqe_tools_original as c_tools
import utils.circuit_tools as circ
import utils.qonversion_tools as qonvert
import utils.linalg_tools as la
from copy import deepcopy
import numpy as np
import itertools
import matplotlib.pyplot as plt

#from qat.interop.qiskit import qiskit_to_qlm
from openfermion.linalg import get_ground_state, jw_configuration_state
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.library import TwoLocal
try:
    from qiskit.utils import QuantumInstance
    from qiskit.algorithms import VQE
except:
    print('Could not import some (non-essential) packages as not installed')
#from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.aqua.components.optimizers import (SLSQP, COBYLA, SPSA, AQGD, L_BFGS_B, P_BFGS,
                                                NELDER_MEAD, CG, ADAM, POWELL, TNC, GSLS,
                                                NFT)#, IMFIL, BOBYQA, SNOBFIT)
from qiskit import Aer
#, NumPyMinimumEigensolver
from qiskit.circuit.library import TwoLocal
#from qiskit.opflow import I, X, Z
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.test.mock import FakeVigo


class cs_vqe_circuit():
    """ Class for constructing and executing CS-VQE Ansatz circuits
    
    Attributes
    ----------
    cs_vqe_results : dict
        Stores CS-VQE output data
    counts : list
        List of optimiser counts
    prmset : list
        List of optimiser parameter settings for each iterate
    values : list
        List of optimiser outputs for each iterate 
    errors : list
        List of VQE errors for each iterate
    ancilla : bool
        Flag indicating whether the circuit requires an ancilla qubit
    red_anz_drop : bool
        Flag indicating whether there is a non-identity operator under projection
    hamiltonian : dict
        Dictionary of Hamiltnonian terms (Pauli strings) and corresponding coefficients
    HF_config: str
        Binary string representing the Hartree-Fock state
    num_qubits : int
        The number of qubits in the Hamiltonian
    G : dict
        Dictionary of commuting noncontextual generators
    A : dict
        Dictionary representing the observable A
    rot_G : bool
        Flag specifying whether to apply G rotations
    rot_A : bool
        Flag specifying whether to apply A rotations
    ham_rotations : list
        List of rotations in form [[angle, operator], ...]
    X_index : int
        Index of the X qubit in A2
    X_qubit : int
        Qubit position of the X qubit in A2
    noncon : float
        The noncontextual ground state energy
    truegs : float
        The true ground state energy of the full Hamiltonian
    order : list
        Qubit inclusion ordering - should be optimal from cs_tools
    ham_reduced : list
        List of dictionaries, each representing a reduced CS-VQE hamiltonian
    init_state : str
        Binary string representing the noncontextual ground state
    index_paulis : dict
        Dictionary of single-qubit Pauli operators and their positions in each term of the observable A
    r1, r2 : int
        Coefficients of the terms in the observable A
    A1, A2 :
        Pauli operators comprising the terms in the observable A

    Methods
    -------
    sim_qubits
    reduced_parity_bits
    lost_parity
    reference_state
    plot_gs_amps
    qubit_map 
    reduce_anz_terms
    reduce_rotations 
    project_anz_term
    gs_check_block
    ref_state_block
    anz_block
    swap_entgl_block
    rot_ham_block
    rot_A_block
    A_eig_block
    parity_cascade_block
    full_uccsd
    cancel_pairs
    build_circuit
    init_params
    store_intermediate_result
    CS_VQE
    run_cs_vqe

    """
    # class variables for storing VQE results
    cs_vqe_results = {}
    counts = []
    prmset = []
    values = []
    errors = []
    # if True adds additional qubit to circuit
    ancilla = False
    red_anz_drop = False
    init_param = np.array([])
    
    def __init__(self, model_data=None, hamiltonian=None, terms_noncon=None, num_qubits=None, hf_config=None, order=None):

        if model_data is not None:
            self.hamiltonian    = model_data["ham"]
            self.num_qubits     = model_data["num_qubits"]
            self.HF_config      = model_data["hf_config"]
            self.HF_energy      = model_data["hf_energy"]
            self.terms_noncon   = model_data["terms_noncon"]
            self.num_tapered    = model_data["num_tapered"]
            self.true_gs_nrg    = model_data["true_gs_nrg"]
            #self.true_gs_vec    = model_data["true_gs_vec"]
            self.ham_rotations  = model_data["ham_rotations"]
            self.noncon         = model_data["noncon"]
            self.truegs         = model_data["truegs"]
            self.G              = model_data["G"]
            self.A              = model_data["A"]
            self.X_index        = model_data["X_index"]
            self.X_qubit        = model_data["X_qubit"]
            self.cs_vqe_energy  = model_data["cs_vqe_energy"]
            self.cs_vqe_errors  = model_data["cs_vqe_errors"]
            self.chem_acc_num_q = model_data["chem_acc_num_q"]
            self.order          = model_data["order"]
            self.ham_reduced    = model_data["ham_reduced"]
            self.init_state     = model_data["reference_state"]

        else:
            if not c_tools.contextualQ_ham(deepcopy(hamiltonian)):
                raise Exception('Hamiltonian is not contextual')
            
            self.HF_config = hf_config
            
            self.hamiltonian = hamiltonian
            self.num_qubits = num_qubits

            # epistricted model and reduced Hamiltonian
            cs = cs_vqe_model(hamiltonian, terms_noncon, num_qubits)#, rot_G, rot_A)
            self.ham_rotations = cs.rotations()
            generators = cs.generators()

            # defined here so only executed once instead of at each call
            self.noncon = cs.gs_noncon_energy()
            self.truegs = cs.true_gs()[0]
            self.G = generators[0]
            self.A = generators[1] 

            self.X_index = list(self.A.keys())[0].find('Z')
            self.X_qubit = num_qubits-1-self.X_index

            ham_noncon = {}
            for t in terms_noncon:
                ham_noncon[t] = hamiltonian[t]

            if order is None:
                heuristic = c_tools.csvqe_approximations_heuristic(hamiltonian, ham_noncon, num_qubits, self.truegs)
                self.cs_vqe_energy = heuristic[1]
                self.cs_vqe_errors = heuristic[2]
                for num_q, e in enumerate(heuristic[2]):
                    if e <= 0.0016:
                        self.chem_acc_num_q = num_q
                        break
                order = heuristic[3]

            self.order, self.ham_reduced = cs.reduced_hamiltonian(order)


    def sim_qubits(self, num_sim_q, complement=False):
        """ Specifies which qubits of the full electronic structure problem are to be simulated
        """
        q_pos_ord = [self.num_qubits-1-q for q in self.order]

        if complement:
            sim_indices = self.order[num_sim_q:]
            sim_qubits = q_pos_ord[num_sim_q:]
        else:
            sim_indices = self.order[0:num_sim_q]
            sim_qubits = q_pos_ord[0:num_sim_q]
        # if ordering left to right
        if sim_indices != []:
            sim_qubits, sim_indices = zip(*sorted(list(zip(sim_qubits, sim_indices)), key=lambda x:x[1]))

        return sim_qubits, sim_indices


    #def reduced_parity_bits(self, num_sim_q):
    #    """ For +1-eigenspace of A restriction, stores the qubit 
    #        positions of Z operators in each term for parity calculation
    #    """
    #    IZ1_red = [q for q in self.index_paulis['Z1'] if q in self.sim_qubits(num_sim_q)[0]]
    #    IZ2_red = [q for q in self.index_paulis['Z2'] if q in self.sim_qubits(num_sim_q)[0]]
    #    return IZ1_red, IZ2_red


    def lost_parity(self, Z_qubits, num_sim_q):
        """ Compensates for the 'lost' Z operators under contextual 
            subspace restriction for parity calculation
        """
        lost_parity = 0

        for i in self.sim_qubits(num_sim_q, complement=True)[0]:
            if i in Z_qubits:
                lost_parity += int(self.reference_state()[self.num_qubits-1-i])

        return lost_parity%2
    

    def reference_state(self, input_ham=None):
        """ Generates a reference state for the Ansatz operator
        """
        reference_bits={}

        if input_ham is None:
            for g in self.G.keys():
                reference_bits[self.num_qubits-1-g.find('Z')] = self.G[g]
            a = list(self.A.keys())[0]
            reference_bits[self.num_qubits-1-a.find('Z')] = self.A[a]
            blank_state = list(np.zeros(self.num_qubits, dtype=int))
            for q in reference_bits.keys():
                index = self.num_qubits-1-q
                if round(reference_bits[q]) == -1:
                    blank_state[index] = 1
            ref_string = ''.join([str(i) for i in blank_state])

        else:
            #ham_q = qonvert.dict_to_QubitOperator(input_ham)
            ham_mat = qonvert.dict_to_WeightedPauliOperator(input_ham).to_matrix()
            gs_vec = la.get_ground_state(ham_mat)[1]
            n_q = len(list(input_ham.keys())[0])
            #gs = get_ground_state(get_sparse_operator(ham_q, n_q).toarray())
            #gs_vec = gs[1]
            amp_list = [abs(a)**2 for a in list(gs_vec)]
            sig_amp_list = sorted([(str(index), a) for index, a in enumerate(amp_list) if a > 0.001], key=lambda x:x[1])
            sig_amp_list.reverse()
            reference_str=bit.int_to_bin(int(sig_amp_list[0][0]), n_q)
            for index, b in enumerate(reference_str):
                reference_bits[n_q-1-index] = int(b)
            ref_string = ''.join([str(i) for i in reference_bits.values()])

        return ref_string


    def plot_cs_vqe_errors(self):
        X = range(0, self.num_qubits+1)
        Y = self.cs_vqe_errors
        Y_compare = [self.noncon-self.truegs]
        for ham in self.ham_reduced[1:]:
            ham_mat = qonvert.dict_to_WeightedPauliOperator(ham).to_matrix()
            gs_vector = la.get_ground_state(ham_mat)
            Y_compare.append(gs_vector[0]-self.truegs)
        self.ham_reduced
        fig, ax = plt.subplots()
        ax.plot(X, Y)
        ax.plot(X, Y_compare)
        ax.hlines(0.0016, 0, self.num_qubits, color='pink')
        ax.set_xlabel('Number of qubits simulated')
        ax.set_ylabel('Error (Ha)')
        plt.show()


    def plot_gs_amps(self, num_sim_q):
        """ Plots a histogram of basis state amplitudes in order of weight
        """
        ham_red = self.ham_reduced[num_sim_q]
        ham_mat = qonvert.dict_to_WeightedPauliOperator(ham_red).to_matrix()
        gs_vector = get_ground_state(ham_mat)[1]
        amp_list = [abs(a)**2 for a in list(gs_vector)]
        sig_amp_list = sorted([(index, a) for index, a in enumerate(amp_list) if a > 0.001], key=lambda x:x[1])
        sig_amp_list.reverse()
        X, Y = zip(*sig_amp_list)
        X = [bit.int_to_bin(x, num_sim_q)+' ('+str(x)+')' for x in X]
        fig, ax = plt.subplots()
        ax.bar(X, Y)
        plt.xticks(rotation=45)
        ax.set_xlabel('Basis state')
        ax.set_ylabel('Probability of measurement')
        plt.show()


    def qubit_map(self, num_sim_q):
        """ Maps the simulated qubits onto those available
        """
        q_map={}
        sim_qubits = self.sim_qubits(num_sim_q)[0]
        for q in sim_qubits:
            try:
                q_map[q] = num_sim_q - 1 - (sim_qubits).index(q)
            except:
                pass    

        return q_map


    def reduce_anz_terms(self, anz_terms, num_sim_q, fix_X_qubit=False):
        """ Very rudimentary restriction - simply deletes qubit 
            positions fixed by noncontextual ground state
        """
        if type(anz_terms)!=dict:
            anz_terms = {t:0 for t in anz_terms}
        #if self.rot_G:
        #    anz_terms = c_tools.rotate_operator(self.rotations, anz_terms)
        
        anz_terms_reduced = {}
        for p in anz_terms.keys():
            blank_op = ['I' for i in range(num_sim_q)]
            for i, sim_i in enumerate(self.sim_qubits(num_sim_q)[1]):
                if fix_X_qubit:
                    if sim_i != self.X_index:
                        blank_op[i] = p[sim_i]
                else:
                    blank_op[i] = p[sim_i]
            
            #if set(blank_op) != {'I'}:
            t = ''.join(blank_op)
            if self.ancilla:
                t = 'I' + t
            if t in anz_terms_reduced.keys():
                anz_terms_reduced[t] = anz_terms_reduced[t] + anz_terms[p]
            else:
                anz_terms_reduced[t] = anz_terms[p]

        return anz_terms_reduced

    
    def reduce_rotations(self, rotations, num_sim_q):
        """ Same method as reduce_anz_terms but for the rotations 
            diagonalising the noncontextual Hamiltonian
        """
        rot = [tuple(r) for r in deepcopy(rotations)]
        rot_red = []
        for r, p in rot:
            blank_op = ['I' for i in range(num_sim_q)]
            for i, sim_i in enumerate(self.sim_qubits(num_sim_q)[1]):
                blank_op[i] = p[sim_i]
            if r == 'pi/2':
                r = np.pi/4
            else:
                r=r/2
            if set(blank_op) != {'I'}:
                t = ''.join(blank_op)
                if self.ancilla:
                    t = 'I' + t
                rot_red.append((t, r))

        return rot_red


    def project_anz_terms(self, anz_terms, num_sim_q):
        """ Determines which Ansatz terms to keep from the full operator for the reduced form
        """
        sim_indices = list(self.sim_qubits(num_sim_q)[1])
        sim_complement = list(set(range(self.num_qubits))-set(sim_indices))
        anz_rot = c_tools.rotate_operator(self.ham_rotations, anz_terms)
        nc_state = self.reference_state()
        proj_anz = {}
        XY_ops = []

        for p in anz_rot.keys():
            proj_pauli = [p[i] for i in sim_complement]
            if ('X' not in proj_pauli) and ('Y' not in proj_pauli):
                parity = int(len([i for i in sim_complement if nc_state[i]=='1' and p[i]=='Z'])%2)
                sgn = 1-2*parity
                
                sim_pauli_list = [p[i] for i in sim_indices]
                sim_pauli = ''.join(sim_pauli_list)
                
                #flp_pauli_list = deepcopy(sim_pauli_list)
                #flp_pauli_list.reverse()
                #flp_pauli = ''.join(flp_pauli_list)
                
                coeff = anz_rot[p]
                if set(sim_pauli) != {'I'}:
                    if sim_pauli in proj_anz.keys():
                        proj_anz[sim_pauli] += sgn*coeff
                    else:
                        proj_anz[sim_pauli] = sgn*coeff
                    #if flp_pauli in proj_anz.keys():
                    #    proj_anz[flp_pauli] += -sgn*coeff
                    #else:
                    #    proj_anz[flp_pauli] = -sgn*coeff
            else:
                t = np.tan(anz_rot[p])
                #sim_pauli_list = [p[i] for i in sim_indices]
                #sim_pauli = ''.join(sim_pauli_list)
                sim_pauli=p
                XY_ops.append((sim_pauli, t))
        
        pair_prod=[]
        for i, P_t in enumerate(XY_ops):
            for j in range(i+1, len(XY_ops)):
                PQ = c_tools.pauli_mult(P_t[0], XY_ops[j][0])
                t1t2 = P_t[1] * XY_ops[j][1]
                pair_prod.append((PQ[0], PQ[1]*t1t2))
                
        #XY_anz_ops = {}
        #for p, coeff in pair_prod:
        #    if set(p) != {'I'}:
        #        if p in XY_anz_ops.keys():
        #            proj_anz[p] += coeff
        #        else:
        #            proj_anz[p] = coeff

        
        prod=''.join(['I' for i in range(self.num_qubits)])
        s_coeff=1
        c_coeff=1
        sgn = 1
        for p, t in XY_ops:
            s_coeff *= np.sin(anz_rot[p])
            c_coeff *= np.cos(anz_rot[p])
            prod, new_sgn = c_tools.pauli_mult(prod, p)
            sgn *= new_sgn

        drop_pauli = ''.join([prod[i] for i in sim_indices])
        #if set(drop_pauli) != {'I'}:
        #    self.red_anz_drop = True
        #    proj_anz[drop_pauli] = 0
        #else:
        #    self.red_anz_drop = False
        return proj_anz#, drop_pauli

    def project_anz_terms_alt(self,anz_terms,num_sim_q):
        """
        """
        sim_indices = list(self.sim_qubits(num_sim_q)[1])
        sim_complement = list(self.sim_qubits(num_sim_q, complement=True)[1])
        proj_anz = {}
        param=0

        for op1 in anz_terms.keys():
            original_param = anz_terms[op1]
            rot_op = c_tools.rotate_operator(self.ham_rotations,{op1:anz_terms[op1]})
            for op2 in rot_op.keys():
                proj_pauli = [op2[i] for i in sim_complement]
                if set(proj_pauli) in [{'I'},{'Z'},{'I', 'Z'}]:
                    
                    parity = proj_pauli.count('Z')%2
                    sgn    = 1-2*parity
                    sim_pauli_list = [op2[i] for i in sim_indices]
                    sim_pauli = ''.join(sim_pauli_list)         
                    
                    if set(sim_pauli)!={'I'}:
                        coeff = sgn*rot_op[op2]
                        ratio = rot_op[op2]/original_param

                        if sim_pauli not in proj_anz:
                            proj_anz[sim_pauli] = {"param":[param], 
                                                   "coeff":[coeff], 
                                                   "ratio":[ratio],
                                                   "initp":[anz_terms[op1]]}

                        else:
                            proj_anz[sim_pauli]["param"].append(param)
                            proj_anz[sim_pauli]["coeff"].append(coeff)
                            proj_anz[sim_pauli]["ratio"].append(ratio)
            
            param += 1
            
        param_lists = set([str(v['param']) for v in proj_anz.values()])
        param_map = {lst:ParameterVector(name='P', length=len(param_lists))[index] for index, lst in enumerate(param_lists)}

        for op in proj_anz.keys():
            param_list = str(proj_anz[op]['param'])
            proj_anz[op]['param'] = param_map[param_list]
            proj_anz[op]['coeff'] = sum(proj_anz[op]['coeff'])
            proj_anz[op]['ratio'] = sum(proj_anz[op]['ratio'])

        return proj_anz

    #def project_anz_terms_alt_2(self,anz_terms,num_sim_q):


    ################################## circuit blocks #################################
    ## Below are circuit components that may be selected in the build_circuit method ##
    ###################################################################################

    def gs_check_block(self, qc, num_sim_q):
        """ Insert as reference the true ground state for development purposes (not to be used in practice)
        """
        ham_red = self.ham_reduced[num_sim_q]
        ham_mat = qonvert.dict_to_WeightedPauliOperator(ham_red).to_matrix()
        gs_vector = la.get_ground_state(ham_mat)[1]
        qc.initialize(gs_vector)


    def ref_state_block(self, qc, num_sim_q, ref_type='nc_gs'):
        """ Inserts a reference state for the Ansatz operator, such as 
            Hartree-Fock or the noncontextual ground state configuration
        """
        q_map = self.qubit_map(num_sim_q)
        sim_qubits = self.sim_qubits(num_sim_q)[0]
        
        if ref_type=='most significant':
            reference_state = self.reference_state(self.hamiltonian)
        elif ref_type=='nc_gs':
            reference_state = self.reference_state()
        elif ref_type=='HF':
            reference_state = self.HF_config
        else:
            raise ValueError('Invalid reference state type - must be "most significant", "nc_gs" or "HF".')
        
        for q in sim_qubits:
            q_index = self.num_qubits-1-q
            if reference_state[q_index] == '1':
                qc.x(q_map[q])


    def anz_block(self, anz_terms, qc, num_sim_q):
        """ Inserts gates effecting the reduced Ansatz operator
        """
        q_map = self.qubit_map(num_sim_q)
        
        if anz_terms is not None:
            anz_terms_reduced = self.project_anz_terms_alt(anz_terms, num_sim_q)
            if anz_terms_reduced == {}:
                # because VQE requires at least one parameter...
                #raise Exception('No terms in the projected Ansatz')
                qc.rz(ParameterVector('P', 1)[0], 0)
            else:
                p_ops=[]
                param=[]
                temp_param=(0,0)
                init_param = []         
                for op, parameters in anz_terms_reduced.items():
                    p_ops.append(op)
                    init_param.append(parameters['coeff'])
                    
                    #if temp_param[0] == parameters['param']:
                    #    ratio = temp_param[1]/parameters['coeff']
                    #    param.append(ratio*parameters['param'])
                    #else:
                    #    param.append(parameters['param'])   
                    #temp_param = (parameters['param'], parameters['coeff'])
                self.init_param = np.array(init_param)
                qc = circ.circ_from_paulis(paulis=p_ops, circ=qc, trot_order=2, dup_param=False)
        else:
            qc += TwoLocal(num_sim_q, 'ry', 'cx', 'full', reps=2, insert_barriers=False)


    def swap_entgl_block(self, qc, num_sim_q):
        """ For restricting to +1-eigenspace of A, fixes the X-qubit 
            to 0 by swapping its entanglement with an ancilla bit
        """
        assert(self.ancilla == True)

        q_map = self.qubit_map(num_sim_q)
        qc.cx(q_map[self.X_qubit], num_sim_q)
        qc.cx(num_sim_q, q_map[self.X_qubit])


    def rot_ham_block(self, qc, num_sim_q, inverse=False):
        """ Performs the rotations diagonalising the commuting noncontextual 
            generators (i.e. excluding unitary partitioning rotations)
        """
        ham_rot = self.reduce_rotations(self.ham_rotations, num_sim_q)
        if inverse:
            ham_rot.reverse()
        for p, r in ham_rot:
            if inverse:
                qc = circ.exp_P(p, circ=qc, rot=-r)
            else:
                qc = circ.exp_P(p, circ=qc, rot=r)


    def rot_A_block(self, qc, num_sim_q, inverse=False):
        """ Determines and subsequently performs the unitary partitioning rotations
        """
        diag_A = []
        rot = c_tools.pauli_mult(self.A1, self.A2)
        t = np.arctan(self.r1/self.r2)
        diag_A.append([t, rot[0]])
        single_A = c_tools.rotate_operator([[t*(rot[1]*1j).real, rot[0]]], self.A)
        Z_indices = [g.find('Z') for g in self.G]
        single_A_indices = [i for i, p in enumerate(list(single_A.keys())[-1]) if p == 'Z']
        ind_Z = [i for i in single_A_indices if i not in Z_indices]
        for i in ind_Z:    
            blank_op = ['I' for i in range(self.num_qubits)]
            blank_op[i] = 'Y'
            rot = ''.join(blank_op)
            diag_A.append(['pi/2', rot])
            rotated = c_tools.pauli_mult(rot, list(single_A.keys())[-1])[0]
            blank_op = ['I' for i in range(self.num_qubits)]
            for j in single_A_indices:
                if rotated[j]=='Z':
                    blank_op[j] = 'Z'
                elif rotated[j] == 'X':
                    blank_op[j] = 'Y'
            rot = ''.join(blank_op)
            diag_A.append(['pi/2', rot])

        A_rot = self.reduce_rotations(diag_A, num_sim_q)
        if inverse:
            A_rot.reverse()
        for p, r in A_rot:
            if inverse:
                qc = circ.exp_P(p, circ=qc, rot=-r)
            else:
                qc = circ.exp_P(p, circ=qc, rot=r)


    def A_eig_block(self, qc, num_sim_q):
        """ Explicitly restrict to the +1-eigenspace of A - requires ancilla
        """
        assert(self.ancilla == True)

        q_map = self.qubit_map(num_sim_q)
        Q = self.r1/(1-self.r2)

        qc.cx(q_map[self.X_qubit], num_sim_q)
        qc.cry(2*np.arctan(Q), num_sim_q, q_map[self.X_qubit])
        qc.x(num_sim_q)
        qc.cry(2*np.arctan(-1/Q), num_sim_q, q_map[self.X_qubit])


    def parity_cascade_block(self, qc, Z_qubits, num_sim_q, inner_circ=None):
        """ Cascade of CNOT gates to compute parity of some subset of qubits and store the result in ancilla 
            qubit. Allows the user to insert a circuit block before uncomputing the parity calculation.
        """
        assert(self.ancilla == True)

        q_map = self.qubit_map(num_sim_q)
        sim_qubits = self.sim_qubits(num_sim_q)[0]
        #index_paulis_reduced = self.index_paulis_reduced(num_sim_q)
        lost_parity = self.lost_parity(Z_qubits, num_sim_q)
        
        if lost_parity:
            qc.x(num_sim_q)
            
        cascade_bits = [q_map[q] for q in Z_qubits if q in q_map.keys()]

        #store parity in ancilla qubit (8) via CNOT cascade
        qc = circ.cascade(cascade_bits+[num_sim_q], circ=qc)

        if inner_circ is None:
            qc.cz(num_sim_q, q_map[self.X_qubit])
        else:
            inner_circ(qc, num_sim_q)
        
        #reverse CNOT cascade
        qc = circ.cascade(cascade_bits+[num_sim_q], circ=qc, reverse=True)

        if lost_parity:
            qc.x(num_sim_q)


    ###################################################################################
    ####################### Circuit builder and VQE functionality #####################
    ###################################################################################


    def full_uccsd(self, anz_terms):
        """ Full UCCSD operator for benchmarking
        """
        q_map = self.qubit_map(self.num_qubits)
        qc = QuantumCircuit(self.num_qubits)
        for q in self.HF_config:
            qc.x(q_map[int(q)])
        qc = circ.circ_from_paulis(paulis=list(anz_terms.keys()), circ=qc, trot_order=2, dup_param=False)
        self.cancel_pairs(circ=qc, hit_set={'s', 'sdg'})
        self.cancel_pairs(circ=qc, hit_set={'h', 'h'})
        return qc


    def cancel_pairs(self, circ, hit_set):
        """ Cancel neighbouring inverse gates, eg. S and Sdg, or H^2
        """
        gate_dict={}
        for i in range(circ.num_qubits):
            gate_dict[i] = []
        for index, gate in enumerate(circ.data):
            gate_dict[gate[1][0].index].append((index, gate[0].name))
        delete_index = []
        for i in gate_dict.keys():
            gates = gate_dict[i]
            for j in range(len(gates)-1):
                pair = list(zip(*(gates[j], gates[j+1])))
                if set(pair[1]) == hit_set:
                    for k in pair[0]:
                        delete_index.append(k)
        delete_index = sorted(delete_index)

        shift=0
        for i in delete_index:
            circ.data.pop(i-shift)
            shift+=1
    

    def build_circuit(self, anz_terms, num_sim_q):
        """ This is where the circuit blocks are selected and the quantum circuit object is built
        """
        self.ancilla = False

        if self.ancilla:
            dim = num_sim_q + 1
        else:
            dim = num_sim_q

        # Initialise the Ansatz
        qc = QuantumCircuit(dim)
        self.ref_state_block(qc, num_sim_q)#, ref_type='HF')
        self.anz_block(anz_terms, qc, num_sim_q)

        # Qiskit variational_algorithm struggling with only one parameter
        #if qc.num_parameters == 1:
        #    qc.rz(Parameter('b'), 0)

        # Cancellation of neighbouring S, Sdg and H gates arising from exponentiation
        self.cancel_pairs(circ=qc, hit_set={'s', 'sdg'})
        self.cancel_pairs(circ=qc, hit_set={'h', 'h'})

        return qc


    def qlm_circuit(self, anz_terms, num_sim_q):
        """Method for generating gate intructions to build CS-VQE circuit on Atos QLM
        """
        qc = self.build_circuit(anz_terms, num_sim_q)
        
        instructions=[]
        param_num=0
        for gate, q_pos, blank in qc.data:
            var = gate.params
            if var != []:
                var = [param_num%qc.num_parameters]
                param_num += 1
            instructions.append((gate.name.upper(), var, [q.index for q in q_pos]))

        qc_qlm_dict = { 'hamiltonian': self.ham_reduced[num_sim_q],
                        'num_qubits':  num_sim_q,
                        'num_params':  qc.num_parameters,
                        'instructions':instructions}

        return qc_qlm_dict


    def init_params(self, anz_terms, num_sim_q):
        """ Returns the initial parameters for input into classical optimiser
        """
        qc = self.build_circuit(anz_terms, num_sim_q)

        if anz_terms is not None:
            anz_red = self.project_anz_terms(anz_terms, num_sim_q)
            if anz_red != {}:
                init_anz_params = np.array([(anz_red[p]).imag for p in anz_red.keys() if set(p)!={'I'}])
                if len(init_anz_params) != qc.num_parameters:
                    init_anz_params = np.append(init_anz_params, 0)  
            else:
                init_anz_params = np.zeros(qc.num_parameters)

        return init_anz_params


    def store_intermediate_result(self, eval_count, parameters, mean, std):
        """ Outputs intermediate data during VQE routine
        """
        self.counts.append(eval_count)
        self.prmset.append(parameters)
        self.values.append(mean)
        self.errors.append(std)


    def CS_VQE(self, anz_terms=None, num_sim_q=None, ham=None, optimizer=POWELL(maxiter=10000), param_bound=np.pi, check_A=False, show_amps=False):
        """ Runs CS-VQE for a given Ansatz operator and number of qubits to simulate
        """
        self.counts.clear()
        self.prmset.clear()
        self.values.clear()
        self.errors.clear()

        if show_amps:
            self.plot_gs_amps(num_sim_q)

        qc = self.build_circuit(anz_terms, num_sim_q)

        #if anz_terms is not None:
        #    anz_red = self.project_anz_terms(anz_terms, num_sim_q)
        #    if anz_red != {}:
        #        init_anz_params = np.array([(anz_red[p]) for p in anz_red.keys() if set(p)!={'I'}])
        #        if len(init_anz_params) != qc.num_parameters:
        #            init_anz_params = np.append(init_anz_params, 0)  
        #    else:
        #        init_anz_params = np.zeros(qc.num_parameters)

        init_anz_params = self.init_param
        while len(init_anz_params) != qc.num_parameters:
            init_anz_params = np.append(init_anz_params, 0)
        bounds = np.array([(p-param_bound, p+param_bound) for p in init_anz_params])
        qc.parameter_bounds = bounds

        seed = 42
        backend = Aer.get_backend('statevector_simulator')
        qi = QuantumInstance(backend=backend, seed_simulator=seed, seed_transpiler=seed)

        # print status update
        sim_qubits = self.sim_qubits(num_sim_q)[0]
        ancilla_string = ''
        num_sim_print = num_sim_q
        if self.ancilla:
            ancilla_string = ' + ancilla'
            num_sim_print += 1

        status = '*   Performing %i-qubit CS-VQE over qubit positions %s ...' % (num_sim_print, str(list(sim_qubits))[1:-1])
        print(status+ancilla_string)

        if ham is None:
            ham = self.ham_reduced[num_sim_q]

        if self.ancilla:
            dim = num_sim_q + 1
            input_ham={}
            for p in ham.keys():
                p_add_q = 'I' + p
                input_ham[p_add_q] = ham[p]
        else:
            dim = num_sim_q
            input_ham = ham
        
        # check expectation value of A (if 1 then in +1 eigenspace)
        # uses BFGS optimizer for speed (not accuracy!)
        vqe = VQE(qc, initial_point=init_anz_params, optimizer=L_BFGS_B(maxiter=1000), quantum_instance=qi)
        A_red = self.reduce_anz_terms(self.A, num_sim_q)
        A_red_q = qonvert.dict_to_WeightedPauliOperator(A_red)
        A_vqe_run = vqe.compute_minimum_eigenvalue(operator=A_red_q)
        A_expct = A_vqe_run.optimal_value
        if check_A:
            print('Expectation value of A:', A_expct)
        
        # simulate the input Hamiltonian

        vqe = VQE(qc, initial_point=init_anz_params, optimizer=optimizer, callback=self.store_intermediate_result, quantum_instance=qi)
        vqe_input_ham = qonvert.dict_to_WeightedPauliOperator(input_ham)
        gs_red = la.get_ground_state(vqe_input_ham.to_matrix())
        target_energy = gs_red[0]
        #target_energy = self.cs_vqe_energy[num_sim_q]
        vqe_run = vqe.compute_minimum_eigenvalue(operator=vqe_input_ham)

        counts = deepcopy(self.counts)
        prmset = deepcopy(self.prmset)
        values = deepcopy(self.values)
        errors = deepcopy(self.errors)

        params={}
        paramsymbols = list(qc.parameters)
        paramsetting = list(zip(*[list(p) for p in prmset]))
        
        for index, p in enumerate(paramsymbols):
            params[p.name] = paramsetting[index]

        # compute target energy for projected hamiltonian
        ham_mat = np.matrix(vqe_input_ham.to_matrix())
        eig_mat = np.matrix(la.eigenstate_projector(A_red, dim))
        ham_proj = eig_mat*ham_mat*eig_mat.H
        proj_energy = la.get_ground_state(ham_proj)[0]

        return {'num_sim_q':num_sim_q,
                'sim_qubits':sim_qubits,
                'result':vqe_run.optimal_value,
                'outdist':[abs(e)**2 for e in vqe_run.eigenstate],
                'target':target_energy,
                'projected_target':proj_energy,
                'noncon':self.noncon, 
                'truegs':self.truegs,
                'A_expct':A_expct,
                #'drop_pauli':drop_pauli,
                'numfev':counts,
                'params':params,
                'energy':values,
                'stddev':errors}

    
    def sufficient_anz_terms(self, anz_terms):
        """
        """
        num_sim_q = self.chem_acc_num_q
        anz_list = []
        error = 1
        anz_ops=[]
        avoid_op = []
        while error >= 0.0016:
            anz_list = []
            for index, op in enumerate(anz_terms.keys()):
                if (op not in avoid_op) and (op not in anz_ops):
                    print('Testing operator %i/%i'%(index+1, len(anz_terms)))
                    test_anz_ops = anz_ops + [op]
                    anz = {op:anz_terms[op] for op in test_anz_ops}
                    anz_red = self.project_anz_terms(anz, num_sim_q)
                    if anz_red == {}:
                        avoid_op.append(op)
                    else:
                        cs_vqe_results = self.CS_VQE(anz_terms=anz, 
                                                     num_sim_q=num_sim_q, 
                                                     optimizer=POWELL(maxiter=10000), 
                                                     param_bound=np.pi)
                        op_error = cs_vqe_results['result']-cs_vqe_results['truegs']
                        anz_list.append((op, op_error))
            add_op, error = sorted(anz_list, key=lambda x:x[1])[0]
            anz_ops.append(add_op)
            
            print(anz_ops)
            print('------------------------------------------------------------')
            print('Error with %i operator(s) in Ansatz = ' % len(anz_ops), error)
            print('------------------------------------------------------------')

        return anz_ops


    def run_cs_vqe(self, anz_terms=None, max_sim_q=None, min_sim_q=0, optimizer=POWELL(maxiter=10000), param_bound=np.pi, iters=1, check_A=False):
        """ Runs multiple CS-VQE routine up to and including the maximum specified number of qubits
        """
        if max_sim_q is None:
            max_sim_q = self.num_qubits
        if max_sim_q>self.num_qubits:
            print('WARNING: specified maximum number of qubits to simulate exceeds total', '\n')
            max_sim_q = self.num_qubits

        rows, cols = la.factor_int(max_sim_q-min_sim_q)
        if rows == 1:
            grid_pos = list(range(cols))
        else:
            grid_pos = list(itertools.product(range(rows), range(cols)))

        self.cs_vqe_results = { 'rows':rows, 
                                'cols':cols,
                                'grid_pos':grid_pos,
                                'noncon':self.noncon, 
                                'truegs':self.truegs, 
                                'num_qubits':self.num_qubits,
                                'X_index':self.X_index}

        for index, grid in enumerate(grid_pos):
            num_sim_q = index+1+min_sim_q
            vqe_iters=[]
            for i in range(iters):
                vqe_run = self.CS_VQE(anz_terms, num_sim_q, optimizer=optimizer,param_bound=param_bound, check_A=check_A)
                vqe_iters.append(vqe_run)
                print('**  Contextual target:', round(vqe_run['target'], 15), '| VQE result:', round(vqe_run['result'], 15))
                error = vqe_run['result']-vqe_run['target']
                if error < 0.0016:
                    print('*** Succesfully converged on CS-VQE target energy')
                    break
                else:
                    print('*** Did not converge on contextual target | Error = ', round(error, 5))
            print(' ')
            vqe_iters = sorted(vqe_iters, key=lambda x:x['result'])
            self.cs_vqe_results[grid] = vqe_iters[0]
