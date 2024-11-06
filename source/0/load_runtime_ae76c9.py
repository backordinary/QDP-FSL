# https://github.com/DanorRon/qiskit-vqe/blob/c71838739c00b4947c8ce50cd0661911211bbada/1d%20TFIM/Real%20Quantum%20Computer/load_runtime.py
import qiskit
from qiskit import IBMQ, QuantumCircuit
from qiskit.providers.ibmq import RunnerResult
import qiskit_ibm_runtime
from qiskit_ibm_runtime.program import UserMessenger, ProgramBackend
from qiskit_ibm_runtime import QiskitRuntimeService
import os
import json

provider = IBMQ.load_account()
backend = provider.get_backend('ibmq_manila')
#can_use_runtime = provider.has_service('runtime')

program_data = os.path.join('/Users/ronanvenkat/Documents/Qiskit VQE/1d TFIM/Real Quantum Computer/vqe_runtime.py')
program_metadata = os.path.join('/Users/ronanvenkat/Documents/Qiskit VQE/1d TFIM/Real Quantum Computer/vqe_runtime.json')

service = QiskitRuntimeService()
#program_id = service.upload_program(data=program_data, metadata=program_metadata)
#service.delete_program(program_id='vqe-runtime-ap4Vvjdr13')