# https://github.com/lukasszz/qiskit-exp/blob/ce14d53735870e7b6ace352629eb4049e9cd6740/h2_lh/h2_e03_from_hdf5.py
"""
This experiment computes the H2 molecule energy starting from provided HDF5 file
First with Exect method then o the quantum simulator.

https://nbviewer.jupyter.org/github/Qiskit/qiskit-tutorial/blob/master/qiskit/aqua/chemistry/dissociation_profile_of_molecule.ipynb

Result:
=== GROUND STATE ENERGY ===

* Electronic ground state energy (Hartree): -1.857271629452
  - computed part:      -1.857271629452
  - frozen energy part: 0.0
  - particle hole part: 0.0
~ Nuclear repulsion energy (Hartree): 0.719968991279
> Total ground state energy (Hartree): -1.137302638173
  Measured:: Num particles: 2.000, S: 0.000, M: 0.00000


"""
from qiskit import Aer
from qiskit_chemistry import QiskitChemistry

qiskit_chemistry_dict = {
    'driver': {'name': 'HDF5'},
    'HDF5': {'hdf5_input': 'H2_equilibrium_0.735_sto-3g.hdf5'},
    'operator': {'name': 'hamiltonian',
                 'qubit_mapping': 'parity',
                 'two_qubit_reduction': True},
    'algorithm': {'name': 'ExactEigensolver'}
}
solver = QiskitChemistry()
result = solver.run(qiskit_chemistry_dict)
print('Ground state energy (classical): {:.12f}'.format(result['energy']))

# Second, we use variational quantum eigensolver (VQE)
qiskit_chemistry_dict['algorithm']['name'] = 'VQE'
qiskit_chemistry_dict['optimizer'] = {'name': 'SPSA', 'max_trials': 350}
qiskit_chemistry_dict['variational_form'] = {'name': 'RYRZ', 'depth': 3, 'entanglement': 'full'}
backend = Aer.get_backend('statevector_simulator')

solver = QiskitChemistry()
result = solver.run(qiskit_chemistry_dict, backend=backend)
print('Ground state energy (quantum)  : {:.12f}'.format(result['energy']))
print("====================================================")
# You can also print out other info in the field 'printable'
for line in result['printable']:
    print(line)
