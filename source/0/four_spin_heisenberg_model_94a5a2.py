# https://github.com/Aqasch/VQE/blob/6b0d3e39f5518a63c3489b8cc1a4696c78998acc/src/four_spin_heisenberg_model.py
#%%
from qiskit import *
from qiskit.aqua.operators import X, Z, Y, I
import numpy as np
import pprint
from qiskit.aqua import QuantumInstance, aqua_globals
from qiskit.aqua.algorithms import VQE, NumPyEigensolver, NumPyMinimumEigensolver
from qiskit.aqua.components.optimizers import SLSQP, COBYLA, CG, POWELL, NELDER_MEAD
from qiskit.circuit.library import TwoLocal
import pprint
#%%
#SET THE COEFFICIENT OF THE HAMILTONIAN:
J = 0.25

#THE HAMILTONIAN:
H = (J * X ^ X ^ I ^ I) + (J * Y ^ Y ^ I ^ I) + \
    (J * Z ^ Z ^ I ^ I) + (J * I ^ X ^ X ^ I) + \
    (J * I ^ Y ^ Y ^ I) + (J * I ^ Z ^ Z ^ I) + \
    (J * I ^ I ^ X ^ X) + (J * I ^ I ^ Y ^ Y) + \
    (J * I ^ I ^ Z ^ Z) + (J * X ^ I ^ I ^ X) + \
    (J * Y ^ I ^ I ^ Y) + (J * Z ^ I ^ I ^ Z)


#%%
optimizers = [SLSQP(maxiter = 200), COBYLA(maxiter = 200), NELDER_MEAD(maxiter = 200)]
converge_cnts = np.empty([len(optimizers)], dtype=object)
converge_vals = np.empty([len(optimizers)], dtype=object)

def hamiltonian_ground_state():
    for i, optimizer in enumerate(optimizers):
        aqua_globals.random_seed = 10
        var_form = TwoLocal(rotation_blocks='ry', entanglement_blocks='cz')
        counts = []
        values = []
        def store_intermediate_result(eval_count, parameters, mean, std):
            counts.append(eval_count)
            values.append(mean)
    
        vqe = VQE(H, var_form, optimizer, callback=store_intermediate_result,
                quantum_instance=QuantumInstance(backend=BasicAer.get_backend('statevector_simulator')))
        result = vqe.compute_minimum_eigenvalue(operator=H)
        true_result = NumPyMinimumEigensolver(operator = H).compute_minimum_eigenvalue()
        converge_cnts[i] = np.asarray(counts)
        converge_vals[i] = np.asarray(values)
        pp = pprint.PrettyPrinter(indent=4)
    
    return values, counts



# if __name__ == '__main__':

    #result['eigenstate'], result['eigenvalue'].real, true_result['eigenvalue'].real     
    #%%

    # x = hamiltonian_ground_state()
    #%%
    # print(x)
    # np.save('spectrum_data/four_spin_heisenberg_hamil_ground_state',hamiltonian_ground_state()[0])
    # print('[NOTE] The ground state has been successfully saved!')

    # #%%
    # print('GROUND EIGENSTATE: ', hamiltonian_ground_state()[0])
    # print('')
    # print('')
    # print('MINIMUM EIGENVALUE (VQE): ', hamiltonian_ground_state()[1])
    # print('')
    # print('')
    # print('MINIMUM EIGENVALUE (TRUE): ', hamiltonian_ground_state()[2])
    # print('')
    # print('')
    # print('PERCENTAGE ERROR IN SIMULATION: {}%'.format(((hamiltonian_ground_state()[2]-hamiltonian_ground_state()[1])/hamiltonian_ground_state()[1])*100))
    # %%
