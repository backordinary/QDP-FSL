# https://github.com/joostvdzalm/Internation-QI-team/blob/ea828091888f8b3a04f818b210297fa587541e93/Quantum%20Process%20Tomography/2qubit_process_tomography_s5/backend_tools.py
from qiskit import Aer
import qiskit as qk

def IBM_backend( ibm_backend ):
    '''
    Backends:
    'ibmq_qasm_simulator',
    'ibmqx2'
    'ibmq_16_melbourne',
    'ibmq_vigo'
    'ibmq_ourense'
    'ibmq_valencia'
    'ibmq_armonk'
    'ibmq_athens'
    'ibmq_santiago'
    '''
    from qiskit import IBMQ
    ibm_token='467524841963e35eb595f88a3884dc21cdb9ac157b26a0d976785344b089cf1979c80e27c9c62f31dec260745c7731eb0a08a714c6969749f2376214ecbfac66'
    ibm_provider = IBMQ.enable_account(ibm_token)
    return  ibm_provider.get_backend(ibm_backend)

    

  



def QI_backend( qi_backend  ):
    '''
    Backends:
    'QX single-node simulator', 
    'Spin-2', 
    'Starmon-5'
    '''
    from quantuminspire.qiskit import QI
    from quantuminspire.credentials import save_account
    qi_token='5697ec173ded946fc884e20d46b523de320a625c'
    save_account(qi_token)
    QI.set_authentication()
    return QI.get_backend(qi_backend) # Possible options: 'QX single-node simulator', 'Spin-2', 'Starmon-5'


def noisy_simulator_backend():
    '''
    Simulator with noise.
    '''
    from qiskit.test.mock import FakeVigo
    return  FakeVigo()

def simulator_backend():
    '''
    Noise free simulator backend.
    '''
    return Aer.get_backend('qasm_simulator')
