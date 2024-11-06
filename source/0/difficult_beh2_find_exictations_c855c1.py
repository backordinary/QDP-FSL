# https://github.com/schraderSimon/masterthesis/blob/3dade3a4a35531c24162b0c10cb4c49b2224f95b/code/systems/quantum_computing/misc/difficult_BeH2_find_exictations.py
from measure_overlap import *
from pyscf import gto, scf, ao2mo, fci, mcscf
import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, cc,scf, ao2mo,fci
np.set_printoptions(linewidth=300,precision=10,suppress=True)
from scipy.linalg import block_diag, eig, orth
from numba import jit
from matrix_operations import *
from helper_functions import *
from scipy.optimize import minimize, root,newton
from qiskit_nature.properties.second_quantization.electronic import ElectronicEnergy, ParticleNumber
from qiskit_nature.properties.second_quantization.electronic.electronic_structure_driver_result import ElectronicStructureDriverResult
from qiskit_nature.properties.second_quantization.electronic.bases import ElectronicBasis, ElectronicBasisTransform

from qiskit_nature.properties import GroupedProperty
import numpy as np
import matplotlib.pyplot as plt
from qiskit.algorithms import NumPyEigensolver,VQE
#from qiskit.aqua.algorithms import VQE
from qiskit_nature.properties.second_quantization.electronic.integrals import (
    ElectronicIntegrals,
    OneBodyElectronicIntegrals,
    TwoBodyElectronicIntegrals,
    IntegralProperty,
)
from qiskit_nature.properties.second_quantization.electronic.bases import ElectronicBasis
from qiskit import Aer, transpile
from qiskit.providers.aer import AerSimulator

from qiskit.algorithms.optimizers import *
from qiskit_nature.circuit.library import UCC,UCCSD, HartreeFock, PUCCD
from qiskit.circuit.library import EfficientSU2, QAOAAnsatz
from qiskit_nature.algorithms import VQEUCCFactory
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper, ParityMapper,BravyiKitaevMapper, BravyiKitaevSuperFastMapper
from qiskit.utils import QuantumInstance
from qiskit.quantum_info.operators import Operator, Pauli
from qiskit.opflow import X, Y, Z, I
from qiskit.opflow import CircuitStateFn, StateFn, TensoredOp
from qiskit.opflow import PauliExpectation, CircuitSampler, StateFn, AerPauliExpectation, MatrixExpectation
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.converters import circuit_to_gate
from qiskit.tools.visualization import circuit_drawer
from qiskit_nature.transformers.second_quantization.electronic import ActiveSpaceTransformer, FreezeCoreTransformer
from qiskit.opflow.primitive_ops import Z2Symmetries
import sys
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
def molecule(x):
    y = lambda x: 2.54 - 0.46*x
    atom="H  " + str(-y(x)) + " 0 " + str(x) + "; H " + str(y(x)) + " 0  " + str(x) + "; Be 0 0 0"
    return atom

basis="STO-6G"
mo_coeff_min=[]
molecular_orbital_energies=[]
mo_coeff_1=[]
enuc=[]
irreps=[]
occdict1={"A1":6,"B1":0,"B2":0}
occdict2={"A1":4,"B1":2,"B2":0}
occdict3={"A1":4,"B1":0,"B2":2}
occdicts=[occdict1,occdict2,occdict3]
xs=x_of_interest=np.array([0,0.1,0.2,0.3,0.4,1,2,3,4])

energies=np.zeros((len(xs),3))
energies_min=np.zeros(len(xs))
active_space=[1,2,3,4,5,6]
nelec=4

for k,x in enumerate(xs):
    print(x)
    atom=molecule(x)
    mol = gto.M(atom=atom, basis=basis, symmetry='C2v', unit='bohr')
    mo_coeff_temp=[]
    mo_en_temp=[]
    for i in [0,1,2]:
        mf = scf.RHF(mol)
        mf.verbose=0
        mf.irrep_nelec=occdicts[i]
        e=mf.kernel(verbose=0)
        mo_coeff_temp.append(mf.mo_coeff)
        mo_en_temp.append(mf.mo_energy)
        energies[k,i]=e
    emindex=np.argmin(energies[k,:])
    irreps.append(occdicts[emindex])
    mo_coeff_min.append(mo_coeff_temp[emindex])


E_exact=np.zeros(len(x_of_interest))
E_UCC=np.zeros(len(x_of_interest))
backend= Aer.get_backend('statevector_simulator')
seed=np.random.randint(100000)
qi = QuantumInstance(backend=backend, seed_simulator=seed, seed_transpiler=seed)

optimizer=SciPyOptimizer("BFGS")#L_BFGS_B()
numPyEigensolver=NumPyEigensolver()

qubit_converter= QubitConverter(mapper=JordanWignerMapper())
optimal_params=None
UCC_1_params=[]
UCC_2_params=[]

for k,x in enumerate(sample_x):
    hamiltonian,num_particles,num_spin_orbitals,nuc_rep,orig_group=get_basis_Hamiltonian(molecule,x,qi,mo_coeff_min[k],basis="STO-6G",active_space=active_space,nelec=nelec,symmetry='C2v',irreps=irreps[k])
    qubit_hamiltonian=qubit_converter.convert(hamiltonian,num_particles=num_particles)
    num_qubits=qubit_hamiltonian.num_qubits
    result = numPyEigensolver.compute_eigenvalues(qubit_hamiltonian)
    E_exact[k]=np.real(result.eigenvalues[0]+nuc_rep)
    print("Start finding Ansatz")
    ansatz,initial_point=SUCC_ansatz(num_particles,num_spin_orbitals,num_qubits,qubit_converter=qubit_converter,reps=2,generalized=False,include_singles=False)
    if optimal_params is None:
        optimal_params=initial_point
    unitary,energy,optimal_params=get_unitary(qubit_hamiltonian,ansatz,optimizer,qi,include_custom=True,initial_point=optimal_params,nuc_rep=nuc_rep)
    UCC_1_params.append(optimal_params)
    E_UCC[k]=np.real(energy)
    print(np.real(energy))
dicterino={}
dicterino["xvals"]=x_of_interest
dicterino["UCC_1"]=UCC_1_params
savemat("SHIT_BeH2_UCC_vals_new.mat",dicterino)
