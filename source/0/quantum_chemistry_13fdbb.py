# https://github.com/benkoehlL/Qiskit_Playground/blob/2a9ede4d0ce3cc9718dccb6ff06538d0ec32824a/quantum_chemistry.py
'''
This program uses some methods from quantum chemistry 
(and doesn't produce anything good yet)
'''

import numpy as np
from matplotlib.pyplot import plot, draw, show
import pylab
from qiskit import *
#from qiskit.chemistry import QiskitChemistry # does no longer exist
from qiskit_nature import *

# input dictionary to configure Qiskit Chemistry for the chemistry problem
qiskit_chemistry_dict = {
    'problem': {'random_seed': 750},
    'driver' : {'name': 'PYSCF'},
    'PYSCF': {'atom': '', 'basis' : 'sto3g'},
    'operator': {'name': 'hamiltonian', 
                'transformation': 'full',
                'qubit_mapping': 'parity',
                'two_qubit_reduction': True},
    'algorithm': {},
}

molecule = 'H .0 .0 -{0}; H .0 .0 {0}'
algorithms = [{'name': 'VQE','operator_mode': 'paulis'},
                {'name': 'ExactEigensolver'}
            ]
optimizer = {'name': 'SPSA', 'max_trials': 200}
variational_form = {'name': 'RYRZ', 'depth': 3, 'entanglement': 'full'}
backend = {'provider': 'qiskit.BasicAer',
            'name': 'qasm_simulator',
             'shots': 1024}
start = 0.5 # start distance
increment = 0.5 # how much to increase distance by
steps = 20 # number of steps to increase by
energies = np.empty([len(algorithms), steps+1])
hf_energies = np.empty(steps+1)
distances = np.empty(steps+1)
print('Processing step __', end='')
for i in range(steps+1):
    print('\b\b{:2d}'.format(i), end='',flush=True)
    d = start + i*increment/steps
    qiskit_chemistry_dict['PYSCF']['atom'] = molecule.format(d/2)

    for j in range(len(algorithms)):
        qiskit_chemistry_dict['algorithm'] = algorithms[j]
        if algorithms[j]['name'] == 'VQE':
            qiskit_chemistry_dict['optimizer'] = optimizer
            qiskit_chemistry_dict['variational_form'] = variational_form
            qiskit_chemistry_dict['backend'] = backend
        else:
            # This does no longer work as qiskit.chemistry is deprecated
            # The functions may be contained in qiskit_nature
            '''
            qiskit_chemistry_dict.pop('optimizer')
            qiskit_chemistry_dict.pop('variational_form')
            qiskit_chemistry_dict.pop('backend')
            solver = QiskitChemistry()
            result = solver.run(qiskit_chemistry_dict)
            energies[j][i] = result['energy']
            hf_energies[i] = result['hf_energy']
            distances[i] = d
            '''
        print(' --- complete')
        print('Distances: ', distances)
        print('Energies: ', energies)
        print('Hartree-Fock energies: ', hf_energies)
pylab.plot(distances, hf_energies, label='Hartree-Fock')
for j in range(len(algorithms)):
    pylab.plot(distances, energies[j], label=algorithms[j])
    pylab.xlabel('Interatomic distance (Angstrom)')
    pylab.ylabel('Energy (Hartree)')
    pylab.title('H2 Ground State Energy')
    pylab.legend(loc='upper right')
    draw()
    show(block=True)        

    pylab.plot(distances, np.subtract(hf_energies, energies[1]), label='Hartree-Fock')
    pylab.plot(distances, np.subtract(energies[0], energies[1]), label=algorithms[0])
    pylab.xlabel('Interatomic distance (Angstrom)')
    pylab.ylabel('Energy (Hartree)')
    pylab.title('Energy difference from ExactEigensolver')
    pylab.legend(loc='upper left')
    draw()
    show(block=True)        


            
            