# https://github.com/QuentinPrigent/quantum-computing-project/blob/499f611cdd9024c8ae424cd60c4e67a7db4a7529/src/services/connection_manager.py
from qiskit import IBMQ, BasicAer, transpile
from qiskit.providers.ibmq import least_busy
from qiskit.tools import job_monitor
import os
from dotenv import load_dotenv


def account_initialization_manager():
    if IBMQ.stored_account() == {}:
        load_dotenv()
        api_key = os.getenv('IBM_API_KEY')
        IBMQ.save_account(api_key)
    IBMQ.load_account()


class ConnectionManager:
    def __init__(self, real_machine=True):
        self.quantum_computer = None
        self.real_machine = real_machine

    def get_least_busy_quantum_computer(self):
        if self.real_machine:
            try:
                provider = IBMQ.get_provider(
                    hub='ibm-q', group='open', project='main'
                )
                self.quantum_computer = least_busy(
                    provider.backends(
                        filters=lambda device: not device.configuration().simulator
                    )
                )
            except Exception:
                print('Impossible to connect to a quantum computer')
        else:
            self.quantum_computer = BasicAer.get_backend('qasm_simulator')
        return self.quantum_computer

    def run_quantum_circuit(self, quantum_circuit, number_of_shots):
        transpiled_quantum_circuit = transpile(quantum_circuit, self.quantum_computer)
        job = self.quantum_computer.run(transpiled_quantum_circuit, shots=number_of_shots)
        job_monitor(job)
        return job
