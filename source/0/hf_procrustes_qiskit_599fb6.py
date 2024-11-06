# https://github.com/schraderSimon/masterthesis/blob/3dade3a4a35531c24162b0c10cb4c49b2224f95b/code/systems/quantum_computing/misc/HF_procrustes_qiskit.py
import matplotlib.pyplot as plt
import numpy as np
import openfermion, cirq
from qiskit import QuantumCircuit, transpile
from measure_overlap import *
def molecule(x):
    y = lambda x: 2.54 - 0.46*x
    atom="H  " + str(-y(x)) + " 0 " + str(x) + "; H " + str(y(x)) + " 0  " + str(x) + "; Be 0 0 0"
    return atom
backend= Aer.get_backend('statevector_simulator')
seed=np.random.randint(100000)
qi = QuantumInstance(backend=backend, seed_simulator=seed, seed_transpiler=seed)

qubit_converter = QubitConverter(mapper=JordanWignerMapper(),two_qubit_reduction=False)

basis="STO-3G"
ref_x=0
new_x=3
mol = mol = gto.M(atom=molecule(ref_x), basis=basis,unit="Bohr")
mol.build()
nelec=mol.nelec
mf = scf.RHF(mol)
mf.kernel()
ref_det=mf.mo_coeff
mo_coeff_min=[]
molecular_orbital_energies=[]
mo_coeff_1=[]
enuc=[]
irreps=[]
occdict1={"A1":6,"B1":0,"B2":0}
occdict2={"A1":4,"B1":2,"B2":0}
occdict3={"A1":4,"B1":0,"B2":2}
occdicts=[occdict1,occdict2,occdict3]
xs=np.linspace(2,3,11)
energies=np.zeros((len(xs),3))
energies_min=np.zeros(len(xs))

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


energies=[]
energies_transformed=[]
HF_energies=[]
HF_energies_transformed=[]
active_space=[1,2,3,4,5,6]
nelec=4
def get_transformation_circuit(R,active_space,nelec):
    n_qubits=len(active_space)
    qubits = cirq.LineQubit.range(n_qubits)
    circuit = cirq.Circuit(openfermion.bogoliubov_transform(qubits,R))
    circuit_qasm=circuit.to_qasm()
    outfile=open("qasm_temp.txt","w")
    outfile.write(circuit_qasm)
    outfile.close()
    circuit_qi=QuantumCircuit.from_qasm_file("qasm_temp.txt")
    return circuit_to_gate(transpile(circuit_qi,optimization_level=3))

for k,x in enumerate(xs):
    procrustes_state,R=localize_procrustes(mol,mo_coeff_min[k],mf.mo_occ,ref_det,mix_states=True,return_R=True,active_orbitals=active_space,nelec=nelec)
    R=np.linalg.inv(mo_coeff_min[k])@procrustes_state
    R=R[1:,1:]
    n_qubits=len(active_space)


    hamiltonian,num_particles,num_spin_orbitals,nuc_rep,orig_group=get_basis_Hamiltonian(molecule,x,qi,mo_coeff_min[k],basis=basis,active_space=active_space,nelec=nelec,symmetry='C2v',irreps=irreps[k])
    qubit_hamiltonian=qubit_converter.convert(hamiltonian)

    ansatz=HartreeFock(num_spin_orbitals=num_spin_orbitals,num_particles=num_particles,qubit_converter=qubit_converter)
    state=CircuitStateFn(ansatz)
    measurable_expression = StateFn(qubit_hamiltonian, is_measurement=True).compose(state)
    expectation = AerPauliExpectation().convert(measurable_expression)
    sampler = CircuitSampler(qi).convert(expectation)
    value=sampler.eval()
    HF_energies.append(value+nuc_rep)
    #result = NumPyEigensolver(k=10).compute_eigenvalues(qubit_hamiltonian)
    #energies.append(np.real(result.eigenvalues[0])+nuc_rep)
    #print(np.real(result.eigenvalues[0])+nuc_rep,np.real(result.eigenvalues[3])+nuc_rep)
    qc=QuantumCircuit(n_qubits*2)
    qc.append(ansatz,list(np.arange(n_qubits*2)))
    gate=get_transformation_circuit(R,active_space,nelec)
    n_qubits=len(active_space)
    qc.append(gate,list(np.arange(n_qubits)))
    qc.append(gate,list(np.arange(n_qubits,2*n_qubits)))
    procrastes_hamiltonian, num_particles,num_spin_orbitals, nuc_rep,newGroup=get_basis_Hamiltonian(molecule,x,qi,procrustes_state,basis=basis,active_space=active_space,nelec=nelec,symmetry='C2v',irreps=irreps[k])
    procrates_qubit_hamiltonian=qubit_converter.convert(procrastes_hamiltonian)
    state1=CircuitStateFn(qc)
    measurable_expression = StateFn(procrates_qubit_hamiltonian, is_measurement=True).compose(state1)
    expectation = AerPauliExpectation().convert(measurable_expression)
    sampler = CircuitSampler(qi).convert(expectation)
    value=sampler.eval()
    HF_energies_transformed.append(value+nuc_rep)
    #result1 = NumPyEigensolver().compute_eigenvalues(procrates_qubit_hamiltonian)
    #energies_transformed.append(np.real(result1.eigenvalues[0])+nuc_rep)
#plt.plot(xs,energies,label="energies")
plt.plot(xs,HF_energies,label="HF")
#plt.plot(xs,energies_transformed,label="transformed")
plt.plot(xs,HF_energies_transformed,label="HF transformed")

plt.legend()
plt.show()
