# https://github.com/lasys/hm_master_thesis/blob/9e605895705534bc0505f19240152c0ebaaf2a30/benchmark/noise_ibm_em/helpers/quantum_instance_helper.py
import qiskit
from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.providers.aer.noise import NoiseModel
from qiskit.ignis.mitigation.measurement import CompleteMeasFitter
from qiskit.providers.models import BackendConfiguration
from .config import *
import pickle 
#
# Quantum Instance 
#
DEFAULT_QASM_SIMULATOR = "qasm_simulator"
NOISE_BACKEND = "ibmq_toronto"

# Noise 
noise_model = None 
coupling_map = None
basis_gates = None

provider = qiskit.IBMQ.get_provider(hub='ibm-q-unibw', group='hochschule-muc', project='masterarbeit')


# create instance depending on config 
def create_quantum_instance():
    from .config import BACKEND

    quantum_instance = None
    
    if BACKEND.value == Backend.Simulator_Noise_Model.value:
        quantum_instance = _create_simulator_with_noise_quantum_instance()
    elif BACKEND.value == Backend.IBMQ_Toronto.value:
        quantum_instance = _create_ibmq_toronto_quantum_instance()
    else: 
        raise Exception(f"Quantum Instance not instanciated! {BACKEND.value}")
    
    return quantum_instance


def _create_simulator_quantum_instance():
    quantum_instance = QuantumInstance(
                           backend=Aer.get_backend(DEFAULT_QASM_SIMULATOR),
                           shots=SHOTS,
                       )
    return quantum_instance

def _init_noise_model_parameters():
    global noise_model, coupling_map, basis_gates
    if noise_model is None or coupling_map is None or basis_gates is None:
        noise_model_filename = f"../helpers/ibmq_toronto_noise_model_dict.pkl"
        noise_model_dict = load_from_pickle(noise_model_filename)
        noise_model = NoiseModel.from_dict(noise_model_dict)
        basis_gates = noise_model.basis_gates
        
        noise_backend_configuration_filename = f"../helpers/ibmq_toronto_noise_backend_configuation_dict.pkl"
        noise_backend_configuration_dict = load_from_pickle(noise_backend_configuration_filename)
        noise_backend_configuration = BackendConfiguration.from_dict(noise_backend_configuration_dict)
        coupling_map = noise_backend_configuration.coupling_map
        

def _create_simulator_with_noise_quantum_instance():
    
    _init_noise_model_parameters()
    
    quantum_instance = QuantumInstance(
                                        backend=Aer.get_backend(DEFAULT_QASM_SIMULATOR),
                                        shots=SHOTS,
                                        noise_model=noise_model,
                                        coupling_map=coupling_map,
                                        basis_gates=basis_gates,
                                        measurement_error_mitigation_cls=CompleteMeasFitter,
                                        measurement_error_mitigation_shots=SHOTS,
                                        )
    return quantum_instance



def load_from_pickle(name):
    with open(name, 'rb') as f:
        return pickle.load(f)
