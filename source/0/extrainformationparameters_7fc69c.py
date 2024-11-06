# https://github.com/oimichiu/quantumGateModel/blob/fa1cb5ed751edbebe512ee299d5484949c856340/IBMQX/qiskit-tutorials/coduriCareNUcompileaza/tutorial_programming_IBM/realQC/1/extraInformationParameters.py
import sys
import os
# Checking the version of PYTHO; we only support > 3.5
if sys.version_info < (3,5):
    raise Exception('Please use Python version 3.5 or greater.')

from pprint import pprint
    
# import qiskit from qiskit-sdk-py folder 
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..', 'qiskit-sdk-py'))
from qiskit import QuantumProgram
import Qconfig

Q_program = QuantumProgram()
Q_program.set_api(Qconfig.APItoken, Qconfig.config['url'])

pprint(Q_program.get_backend_parameters('ibmqx2'))

pprint(Q_program.get_backend_parameters('ibmqx4'))

pprint(Q_program.get_backend_parameters('ibmqx5'))