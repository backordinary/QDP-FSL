# https://github.com/ASU-KE/QuantumCollaborativeSamples/blob/0469e2d73cbcc0b3af53bbfcd4644bafc8bdd09b/2-test_access.py
from qiskit import IBMQ

# Assumes that credentials are stored in $HOME/.qiskit/qiskitrc - see 1-store_credentials.py
IBMQ.load_account()

# Select the ASU Quantum Hub Provider (Hub, Group, and Project) - change PROJECT to your assigned project
provider = IBMQ.get_provider(hub="ibm-q-asu", group="main", project="PROJECT")

# Get and print available backends in the Project
ibmq_backends = provider.backends()
print("Remote backends: ", ibmq_backends)
