# https://github.com/schraderSimon/masterthesis/blob/3dade3a4a35531c24162b0c10cb4c49b2224f95b/code/systems/quantum_computing/misc/setUpProperties.py
import numpy as np
import sys
from qiskit_nature.properties import GroupedProperty
from qiskit_nature.properties.second_quantization.electronic.integrals import (
    ElectronicIntegrals,
    OneBodyElectronicIntegrals,
    TwoBodyElectronicIntegrals,
    IntegralProperty,
)
from qiskit_nature.properties.second_quantization.electronic import (
    ElectronicEnergy,
    ElectronicDipoleMoment,
    ParticleNumber,
    AngularMomentum,
    Magnetization,
)
from qiskit.providers.aer import StatevectorSimulator
from qiskit.utils import QuantumInstance
from qiskit_nature.algorithms import VQEUCCFactory
from qiskit import Aer
from qiskit_nature.drivers import UnitsType, Molecule
from qiskit_nature.drivers.second_quantization import (
    ElectronicStructureDriverType,
    ElectronicStructureMoleculeDriver,
)
from qiskit_nature.problems.second_quantization import ElectronicStructureProblem
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper
from qiskit_nature.algorithms import GroundStateEigensolver

from qiskit_nature.properties.second_quantization.electronic.bases import ElectronicBasis

from pyscf import gto, scf, ao2mo
mol = mol = gto.M(atom='H 0 0 0; H 0 0 1.1', basis='STO-3G',unit="Bohr")
mol.build()
mf = scf.RHF(mol)
mf.kernel()
hcore_ao = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
hcore_mo = np.einsum('pi,pq,qj->ij', mf.mo_coeff, hcore_ao, mf.mo_coeff)
eri_ao = mol.intor('int2e')
eri_mo = ao2mo.incore.full(eri_ao, mf.mo_coeff)



one_body_ints = hcore_mo
two_body_ints = eri_mo
electronic_energy = ElectronicEnergy.from_raw_integrals(
    ElectronicBasis.MO, one_body_ints, two_body_ints
)
particle_number = ParticleNumber(
    num_spin_orbitals=4,
    num_particles=(1, 1),
)
hamiltonian = electronic_energy.second_q_ops()[0]
props=GroupedProperty("molly")
props.add_property(particle_number)
props.add_property(electronic_energy)

qubit_converter = QubitConverter(mapper=JordanWignerMapper())
qubit_op = qubit_converter.convert(hamiltonian)
print(qubit_op)



quantum_instance = QuantumInstance(backend=Aer.get_backend("aer_simulator_statevector"))
vqe_solver = VQEUCCFactory(quantum_instance)
calc = GroundStateEigensolver(qubit_op, vqe_solver)

res = calc.solve(second_q_ops=qubit_op)

print(res)
