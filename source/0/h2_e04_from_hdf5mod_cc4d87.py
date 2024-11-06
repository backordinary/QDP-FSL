# https://github.com/lukasszz/qiskit-exp/blob/ce14d53735870e7b6ace352629eb4049e9cd6740/h2_lh/h2_e04_from_hdf5mod.py
"""
This experiment computes the H2 molecule energy starting from modified HDF5.
The modification is made to one body integral (mo_onee_ints), it was changed to -1.1.
It was made by the HDF5View (https://www.hdfgroup.org/downloads/hdfview/)

Results:
=== GROUND STATE ENERGY ===

* Electronic ground state energy (Hartree): -1.549384951393
  - computed part:      -1.549384951393
  - frozen energy part: 0.0
  - particle hole part: 0.0
~ Nuclear repulsion energy (Hartree): 0.719968991279
> Total ground state energy (Hartree): -0.829415960114
  Measured:: Num particles: 2.000, S: 0.000, M: 0.00000

"""
from qiskit import Aer
from qiskit_chemistry import QiskitChemistry

qiskit_chemistry_dict = {
    'driver': {'name': 'HDF5'},
    'HDF5': {'hdf5_input': 'H2_mod.hdf5'},
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
