# https://github.com/QAMP-Fall-22-Project34/HamiltonianLearning-HubbardDimerEvolve/blob/65120f7c2b57b2f733c1eb1568cc50b7cd5e6c6c/modules/LatticeSolver.py
import sys
from typing import List
from logging import root
from math import pi
import numpy as np
import scipy.linalg as sl
from qiskit import QuantumCircuit
from qiskit_nature.problems.second_quantization.lattice import (
    BoundaryCondition,
    Lattice,
    LatticeDrawStyle,
    LineLattice, 
    FermiHubbardModel
)
from qiskit_nature.operators.second_quantization import FermionicOp
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper
from qiskit_nature.circuit.library import UCC
from Lattices import DimerLattice, Vector
from cVQD import VQD

class LatticeSolver():
    def __init__(self, lattice: DimerLattice ) -> None:
        #Initialize wave function ansatz
        self.lat = lattice
        self.nqubits = lattice.nsites * 2
        init_qc = QuantumCircuit(self.nqubits)
        n_up = int(np.ceil(lattice.num_particles/2))
        n_dn = int(np.floor(lattice.num_particles/2))
        assert self.nqubits >= n_up + n_dn 
        for ie in range(n_up):
            iq = 2*ie
            init_qc.x(iq)
        for ie in range(n_dn):
            iq = self.nqubits - (2*ie + 1)
            init_qc.x(iq)
        #instantiate a generalized UCCGSD ansatz circuit with 2 Trotter steps.
        self.wf_anz = UCC(excitations='sd', qubit_converter=lattice.jw_converter, num_particles=(n_up,n_dn), 
                       num_spin_orbitals=self.nqubits, reps=2, initial_state=init_qc, 
                       preserve_spin=False, generalized=True)
    
    def classical_solve(self, Ops: Vector) -> List:
        #Ops [ Hamiltonian, ParticleNumber, ...]
        assert len(Ops) >= 2  

        HM = Ops[0].to_matrix()
        #Diagonalize
        w, Psi = sl.eig(HM)
        #calculate matrix elements for number operator to select relevant state
        NM = Ops[1].to_matrix()
        N_ev = np.dot(Psi.conj().T,np.dot(NM, Psi))
        expect_vals=[]
        wfns_out=[]
        for j in np.argsort(w):
            j_row = [j]
            if np.abs(N_ev[j,j] - self.lat.num_particles) < 0.1:
                psi_j = Psi[:,j]
                for qop in Ops:
                    if type(qop) == list:
                        qop_ev = [ np.dot(psi_j.conj().T,np.dot(op.to_matrix(), psi_j)) for op in qop ]
                    else:
                        qop_ev = np.dot(psi_j.conj().T,np.dot(qop.to_matrix(), psi_j))
                    j_row.append(np.real(qop_ev))
                expect_vals.append(j_row)
                wfns_out.append(psi_j)
        return np.array([expect_vals, wfns_out],dtype=object)


    def quantum_solve(self, Ops: Vector) -> List:
        vqd = VQD([Ops[0]], self.wf_anz, 0, states=np.random.rand(len(self.wf_anz.preferred_init_points),1))
        (eigenvalues, result_states) = vqd.calculate()
        return (eigenvalues, result_states)