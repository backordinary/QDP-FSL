# https://github.com/jason-jk-kang/QIS-qchem/blob/6f296a4272d842a136892e33f8edd77d2051fc60/Archive/qiskit/2test.py
# useful additional packages
import matplotlib.pyplot as plt
import numpy as np
from qiskit import Aer
from qiskit_chemistry import QiskitChemistry
import warnings
warnings.filterwarnings('ignore')

# setup qiskit_chemistry logging
import logging
from qiskit_chemistry import set_qiskit_chemistry_logging
set_qiskit_chemistry_logging(logging.ERROR) # choose among DEBUG, INFO, WARNING, ERROR, CRITICAL and NOTSET

molecule = 'H .0 .0 .0; H .0 .0 0.74; H .0 .0 3.3'


# First, we use classical eigendecomposition to get ground state energy (including nuclear repulsion energy) as reference.
qiskit_chemistry_dict = {
    'driver': {'name': 'PYSCF'},
    'PYSCF': {'atom': molecule,
              'basis': 'sto3g'},
    'operator': {'name':'hamiltonian',
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
qiskit_chemistry_dict['variational_form'] = {'name': 'RYRZ', 'depth': 3, 'entanglement':'full'}
backend = Aer.get_backend('statevector_simulator')

solver = QiskitChemistry()
result = solver.run(qiskit_chemistry_dict, backend=backend)
print('Ground state energy (quantum)  : {:.12f}'.format(result['energy']))
print("====================================================")
# You can also print out other info in the field 'printable'
for line in result['printable']:
    print(line)
