# https://github.com/Tomoya-Takanashi/ITUCC/blob/ed1c39b6565acd9b176a72f6036c11e382af9161/itmol.py
import sys
sys.path.append('../')
print(sys.path)
from openfermion.chem import MolecularData
from openfermionpyscf import prepare_pyscf_molecule
from openfermionpyscf._run_pyscf import compute_scf
from openfermionpyscf import run_pyscf
from qiskit import Aer,aqua,execute,QuantumCircuit
from qiskit.visualization.utils import _validate_input_state
from math import pi,degrees,acos,sqrt,cos
from joblib import Parallel,delayed
import scipy.optimize
import numpy
import cmath
import vqe


class RTMOL:
    def __init__(self,molecule,
                 time_s = None,
                 delta_t=None,
                 t_var=None):
        au_time = 2.418e-17
        if delta_t is None:
            self.d_t = 0.1e-18/au_time
        else:
            self.d_t = delta_t

        if t_var is None:
            self.t_var = 1e-16/au_time
        else:
            self.t_var = t_var 

        if t_var is None:
            self.time_s = 0
        else:
            self.time_s = time_s

        self.opf_mol = molecule
        self.scf_mol = prepare_pyscf_molecule(molecule)
        self.ele_num = self.opf_mol.get_n_alpha_electrons()+self.opf_mol.get_n_beta_electrons()
        self.aodip = self.scf_mol.intor_symmetric('int1e_r')
        self.hf_mol = compute_scf(self.scf_mol)
        self.hcore = self.hf_mol.get_hcore()
        pyscf_mol = run_pyscf(molecule)
        self.mo_a = pyscf_mol.canonical_orbitals
        self.one_integral,self.two_integral = pyscf_mol.get_integrals()
        self.qubit_num = 2*self.one_integral.shape[0]
        self.g_jw_hamil = vqe.hamiltonian.compute_spin_integral(pyscf_mol,self.mo_a,self.mo_a,self.hcore)
        self.g_jw_opr, self.g_jw_num = vqe.hamiltonian.hamiltonian_grouping(list(self.g_jw_hamil.terms.keys()),
                                                                        list(self.g_jw_hamil.terms.values()))
        self.s_comb,self.d_comb = vqe.anzats.amp_comb.combination(self.ele_num,self.qubit_num)
        self.L_param = len(self.s_comb)+len(self.d_comb)
        self.L_s_param = len(self.s_comb)
        self.L_d_param = len(self.d_comb)

        self.vqe_parameter = vqe.anzats.uccsd_trotter1_param.mp2(self.ele_num,self.qubit_num,self.opf_mol,
                                                                 self.mo_a,self.mo_a,self.hcore)
        self.d_vector = [1,0,0]
        self.omega = 0
        self.s_gate = [['H','Y'],['Y','H']]
        self.d_gate = [['Y','Y','H','Y'],['Y','Y','Y','H'],['H','Y','H','H'],['Y','H','H','H'],
                       ['H','H','H','Y'],['H','Y','Y','Y'],['Y','H','Y','Y'],['H','H','Y','H']]

        self.machine = Aer.get_backend('statevector_simulator')
        
    def get_HF_result(self):
        hf_keys =['replusion','hf_energy','orbital energy']
        hf_values = [self.scf_mol.energy_nuc(),self.scf_mol.e_tot,self.scf_mol.mo_energy]
        return  dict(zip(hf_keys, hf_values))

    def runVQE(self,use_core=None):
        if use_core is None:
            ncpu = 1
        else:
            ncpu = use_core

        e_num = self.ele_num
        q_num = self.qubit_num
        v_param = self.vqe_parameter
        cost_opr = self.g_jw_opr
        cost_num = self.g_jw_num

        def cost(phi):
            vqe_state = 0
            vector = 0
            vqe_state = vqe.reference_wavefunction.HF_state.state(self.ele_num,self.qubit_num)
            vector = vqe.anzats.uccsd_Trotter1.circuit(self.ele_num,self.qubit_num,vqe_state,phi)
            def hamil_cost(hamil_num,ucc_wave):
                return vqe.measurement.statevector_measure.expect(ucc_wave,self.g_jw_opr[hamil_num],self.g_jw_num[hamil_num],
                                                                  self.qubit_num,hamil_num,len(self.g_jw_opr))
            hamil_processed = Parallel(n_jobs=ncpu,backend="threading",verbose=1)([delayed(hamil_cost)(i,vector) for i in range(len(self.g_jw_opr))])
            #print(sum(hamil_processed))
            return sum(hamil_processed)

        res = scipy.optimize.minimize(cost,self.vqe_parameter,method = 'BFGS',options = {"disp": False,"gtol": 1e-4,'maxiter':100,'eps':1e-5})
        vqe_keys = ['opt_param','energy']
        vqe_values = [res.x,res.fun+self.scf_mol.energy_nuc()]
        return dict(zip(vqe_keys,vqe_values))

    def energy_cal(self,phi,use_core=1):
        vqe_state = 0
        vector = 0
        vqe_state = vqe.reference_wavefunction.HF_state.state(self.ele_num,self.qubit_num)
        vector = vqe.anzats.uccsd_Trotter1.circuit(self.ele_num,self.qubit_num,vqe_state,phi)
        def hamil_cost(hamil_num,ucc_wave):
            return vqe.measurement.statevector_measure.expect(ucc_wave,self.g_jw_opr[hamil_num],self.g_jw_num[hamil_num],
                                                              self.qubit_num,hamil_num,len(self.g_jw_opr))
        hamil_processed = Parallel(n_jobs=use_core,backend="threading",verbose=0)([delayed(hamil_cost)(i,vector)
                                                                    for i in range(len(self.g_jw_opr))])
        return sum(hamil_processed)

    
    def get_rt_hamiltonian(self,t): 
        force,power = self.E_field_ver2(t)
        rt_hamiltonian = vqe.hamiltonian.compute_spin_integral(self.opf_mol,self.mo_a,self.mo_a,self.hcore+force)
        hamiltonianListOpr = list(rt_hamiltonian.terms.keys())
        hamiltonianListNum = list(rt_hamiltonian.terms.values())
        return hamiltonianListOpr, hamiltonianListNum, power 

    def E_field_ver1(self,t):
        ans =  numpy.exp( (-1) * ((t / self.t_var)**2) ) * numpy.exp(-1j * self.omega / 27.2114 * t) / sqrt(pi) / self.t_var
        E = []
        E.append(self.aodip[0] * ans * self.d_vector[0])
        f = sum(E)
        E.append(self.aodip[1] * ans * self.d_vector[1])
        f = sum(E)
        E.append(self.aodip[2] * ans * self.d_vector[2])
        f = sum(E)
        
        return f,ans
    
    def E_field_ver2(self,t):
        if abs(t-self.t_var) < self.t_var:
            ans = (1e-3)*(cos((pi/(2*self.t_var))*(t-self.t_var))**2)
        else:
            ans = 0
        E = []
        E.append(self.aodip[0] * ans * self.d_vector[0])
        E.append(self.aodip[1] * ans * self.d_vector[1])
        E.append(self.aodip[2] * ans * self.d_vector[2])
        f = sum(E)
        return f,ans

"""
basis = "sto-3g" #basis set
multiplicity = 1 #spin multiplicity
charge = 0   #total charge for the molecule
#geometry = [("H",(-dis,0,0)),("H",(0,0,0)),("H",(dis,0,0))]
geometry = [("H",(0.37,0,0)),("H",(-0.37,0,0))]
molecule = MolecularData(geometry, basis, multiplicity, charge)

mol = RTMOL(molecule)
print(mol.runVQE())
"""
