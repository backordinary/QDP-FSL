# https://github.com/oimichiu/quantumGateModel/blob/fa1cb5ed751edbebe512ee299d5484949c856340/IBMQX/qiskit-tutorials/coduriCareNUcompileaza/tutoriale-QISKit/reference/approximateQuantumComputers/quantum_chemestry.py
# Quantum chemestry
import sys
import os

# import qiskit from qiskit-sdk-py folder
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../../', 'qiskit-sdk-py'))
    import Qconfig
    qx_config = {
        "APItoken": Qconfig.APItoken,
        "url": Qconfig.config['url']}
except:
    qx_config = {
        "APItoken":"da2a4002660558a35103a600bcbda7fe438cea629a6be98969ea5e367c091b6815e624bd86b6207121bd97fef79c22033318a4402eeafcbd04b021fd80f5a195",
        "url":"https://quantumexperience.ng.bluemix.net/api"
    }

# useful packages
import matplotlib.pyplot as plt 
import numpy as np 
from scipy import linalg as la 
from functools import partial

# importing the QISKit
from qiskit import QuantumProgram 
import Qconfig

# import basic plot tools
from qiskit.tools.visualization import plot_histogram

# import optimization tools
from qiskit.tools.apps.optimization import trial_circuit_ryrz, SPSA_optimization, SPSA_calibration
from qiskit.tools.apps.optimization import Hamiltonian_from_file, make_Hamiltonian
from qiskit.tools.apps.optimization import eval_hamiltonian, group_paulis

# Ignore warnings due to chopping of small imaginary part of the energy
import warnings
warnings.filterwarnings('ignore')

n = 2
m = 6
device = 'local_qasm_simulator'

# Optimization
Q_program = QuantumProgram()
Q_program.set_api(Qconfig.APItoken,Qconfig.config["url"])

initial_Theta = np.random.randn(2*n*m)
entangler_map = Q_program.get_backend_configuration(device)['coupling_map'] # the map of two-qubit gates with control at key and target at values
if entangler_map == 'all-to-all':
    entangler_map = {i: [j for j in range(n) if j !=i] for i in range(n)}
shots = 1
max_trials = 100
ham_name='H2/H2Equilibrium.txt'
# ham_name='LIH/LiHEquilibrium.txt' # For optimization of LiH at bond length
    
# Extract Energy
pauli_list = Hamiltonian_from_file(ham_name)
H=make_Hamiltonian(pauli_list)
exact=np.amin(la.eig(H)[0]).real
print('The exact ground state energy is:')
print(exact)
pauli_list_grouped=group_paulis(pauli_list)


def cost_function(Q_program,H,n,m,entangler_map,shots,device,theta):
    
    return eval_hamiltonian(Q_program,H,trial_circuit_ryrz(n,m,theta,entangler_map,None,False),shots,device).real


initial_c=0.01
target_update=2*np.pi*0.1
save_step = 20

if shots ==1:
    SPSA_params=SPSA_calibration(partial(cost_function,Q_program,H,n,m,entangler_map,shots,device),initial_theta,initial_c,target_update,25)
    output=SPSA_optimization(partial(cost_function,Q_program,H,n,m,entangler_map,shots,device),initial_theta,SPSA_params,max_trials,save_step,1);
else:
    SPSA_params=SPSA_calibration(partial(cost_function,Q_program,pauli_list_grouped,n,m,entangler_map, shots,device),initial_theta,initial_c,target_update,25)
    output=SPSA_optimization(partial(cost_function,Q_program,pauli_list_grouped,n,m,entangler_map,shots,device), initial_theta,SPSA_params,max_trials,save_step,1);

