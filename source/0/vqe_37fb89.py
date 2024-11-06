# https://github.com/dmiracle/qiskit-test/blob/b1a3404316253a0ce06e48d1199618d75958cf62/vqe.py
from qiskit import IBMQ
from qiskit.providers.basicaer import QasmSimulatorPy 


IBMQ.load_account()
IBMQ.providers()
provider = IBMQ.get_provider(hub='strangeworks-hub', group='science-team', project='science-test')
backend = provider.get_backend('ibmq_montreal')