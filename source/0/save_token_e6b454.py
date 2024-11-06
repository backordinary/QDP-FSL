# https://github.com/rdcunha/las-qpe/blob/3a598dcfed5dab26bc8a6c5e033ddcbeb6c0f075/qiskit_calcs/save_token.py
from qiskit import IBMQ
from qiskit.providers.ibmq import least_busy

IBMQ.load_account() # Load account from disk
print(IBMQ.providers()  )  # List all available providers
provider = IBMQ.get_provider(hub='ibm-q')
print(provider.backends())

small_devices = provider.backends(filters=lambda x: x.configuration().n_qubits == 5
                                   and not x.configuration().simulator)
print(least_busy(small_devices))
