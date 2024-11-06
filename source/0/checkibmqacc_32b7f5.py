# https://github.com/takehuge/Qalgorithm/blob/3f26c624ff38e03e915b3949d583cb747d73aca3/QisKit/CheckIBMQacc.py
import qiskit
from qiskit import IBMQ

# Load account from disk:
provider = IBMQ.load_account()
# list the account currently in the session
print(IBMQ.active_account())
# list all available providers
print(IBMQ.providers())
print(provider.backends())

