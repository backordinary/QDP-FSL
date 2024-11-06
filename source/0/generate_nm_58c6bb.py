# https://github.com/rum-yasuhiro/experiments_crosstalk_multitasking/blob/5d92c40dcfc68fab7922e806ab75e898bf3bda81/xtalk_noisemodel/generate_nm.py
from typing import Optional

from qiskit.providers.aer.noise import NoiseModel
import qiskit.providers.aer.noise as noise

from qiskit.providers.ibmq import IBMQBackend
from qiskit.test.mock import FakeBackend
from .util import choose_fakebackend


def generate_nm(noise_model=None, backend=None, xtalk_prop=None):
    """
    args: 
        noise_model: NoiseModel
        backend: backend_name(str) or IBMQBackend or FakeBackend
        xtalk_prop: dict
            e.g. {(0, 1): {(2, 3): 1.2, (3, 4): 2.7}}
    return:
        NoiseModel
    """

    nm = noise_model

    # noise_model from backend
    if not nm: 
        # define backend
        if isinstance(backend, str):
            backend = choose_fakebackend(backend)
        elif isinstance(backend, FakeBackend): 
            pass
        elif isinstance(backend, IBMQBackend):
            pass
        else:
            raise BackendError("backend is not correct")
        nm = NoiseModel.from_backend(backend)

    


def _nonlocal_nm(noise_model, xtalk_prop): 
    # crosstalk property
    if xtalk_prop: 
        return _xtalk_errors(noise_model, xtalk_prop)
    else: 
        return noise_model

def _xtalk_errors(noise_model, xtalk_prop): 

    for qubits, xtalks in xtalk_prop.items(): 
        for noise_qubits, xtalkratio in xtalks.items():
            
            prob = backend.properties().gate_error('cx', noise_qubits)
            xtalk_prob = prob * xtalkratio
            error = noise.depolarizing_error(xtalk_prob, 2) # defined as depolarizing error but fix it later
            nm.add_nonlocal_quantum_error(
                                            error=error, 
                                            instructions='cx', 
                                            qubits=list(qubits), 
                                            noise_qubits=list(noise_qubits),
                                            )

    return nm

class BackendError(Exception): 
    pass