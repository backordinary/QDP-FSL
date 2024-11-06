# https://github.com/compmolsci/Hubbard_model/blob/6a1525bf6f01a125a4e929e5e2f49b80f7d13511/mol_hamilt_qiskit_template.py
#from qiskit.chemistry.drivers import PySCFDriver, UnitsType, Molecule
from qiskit.chemistry.drivers import PySCFDriver, UnitsType, Molecule
from qiskit.chemistry.drivers import HFMethodType
#from qisit.chemistry.core import qmolecule
from qiskit.chemistry import FermionicOperator
from qiskit import QuantumCircuit
#driver=PySCFDriver(atom='H .0 .0 .0; H .0 .0 .7414', unit=UnitsType.ANGSTROM,charge=0,spin=0, basis='sto3g',hf_method=HFMethodType.RHF)
#molecule=driver.run()
#ferop=FermionicOperator(h1=molecule.one_body_integrals,h2=molecule.two_body_integrals)

#print(ferop.h1)
#print(ferop.h2)
dist
molecule = Molecule(geometry=[['H', [0., 0., 0.]],['H', [0., 0., dist]]],charge=0, multiplicity=1)
#molecule = qmolecule(geometry=[['H', [0., 0., 0.]],
#                                  ['H', [0., 0., 0.735]]],
#                                                       charge=0, multiplicity=1)
driver = PySCFDriver(molecule = molecule, unit=UnitsType.ANGSTROM, basis='sto3g')


from qiskit.chemistry.transformations import (FermionicTransformation,FermionicTransformationType,FermionicQubitMappingType)

fermionic_transformation = FermionicTransformation(transformation=FermionicTransformationType.FULL,qubit_mapping=FermionicQubitMappingType.JORDAN_WIGNER,two_qubit_reduction=False,freeze_core=False)

qubit_op, _ = fermionic_transformation.transform(driver)
#n=9
#print(molecule)
#print(qubit_op)

#ckt=QuantumCircuit(4)
#print(qubit_op)
#ckt=qubit_op.to_circuit
#qc=QuantumCircuit(4)
#qc.combine(qubit_op)
#print(type(ckt))
#a=qubit_op[n].exp_i
#print(a)
#print(ckt)
#print(qubit_op[n].coeff, qubit_op[n].primitive)
#print(fermionic_transformation.molecule_info)

'''from openfermion import *
def get_molecule(bond_len):
    #geometry=[(, (0.,0.,0.)), ('H',(0,0,bond_len))]
    description=format(bond_len)
    molecule=MolecularData(geometry, 'sto-3g', 1,description=description)
    molecule.load()
    return molecule

m=get_molecule(0.7414)
Hmol=m.get_molecular_hamiltonian()
#nuclear_repulsion_energy = molecule.nuclear_repulsion_energy
#print(nuclear_repulsion_energy)
#print(Hmol)
hjw=jordan_wigner(get_fermion_operator(m.get_molecular_hamiltonian()))
print(hjw)'''





'''driver = PySCFDriver(atom='H .0 .0 .0; H .0 .0 0.735',charge=0, unit=UnitsType.ANGSTROM, basis='sto3g')
molecule=driver.run()
#driver = PySCFDriver(molecule = molecule, unit=UnitsType.ANGSTROM, basis='sto3g')
from qiskit.chemistry import FermionicOperator
num_particles = molecule.num_alpha + molecule.num_beta
num_spin_orbitals = molecule.num_orbitals * 2

# Build the qubit operator, which is the input to the VQE algorithm in Aqua
ferOp = FermionicOperator(h1=molecule.one_body_integrals, h2=molecule.two_body_integrals)
map_type = 'JORDAN_WIGNER'
qubitOp = ferOp.mapping(map_type)
#qubitOp = Z2Symmetries.two_qubit_reduction(qubitOp, num_particles)
num_qubits = qubitOp.num_qubits
print(ferOp)
#fermionic_transformation = FermionicTransformation( transformation=FermionicTransformationType.FULL, qubit_mapping=FermionicQubitMappingType.JORDAN_WIGNER,two_qubit_reduction=False,freeze_core=False)
#qubit_op, _ = fermionic_transformation.transform(driver)
#print(qubit_op)
#print(fermionic_transformation.molecule_info)'''
