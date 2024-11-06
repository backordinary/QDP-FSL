# https://github.com/junmendoza/QuantumPlayground/blob/903e2b61268810b05674d3e2a5c1fb9b41f38afe/Algorithms/QiskitUtils.py


# Setting up the IBM Qiskit account
from qiskit import IBMQ

def setupQiskit(token) :
    token_present = len(token) > 0;
    qiskit_url = 'https://api.quantum-computing.ibm.com/api/Hubs/ibm-q/Groups/open/Projects/main'
    
    # Need to migrate from v1 to v2
    # May have to just deprecate this altogether
    migrate_v1_to_v2 = False
    if migrate_v1_to_v2 :
        IBMQ.update_account()
    
    store_acct_to_disk = False
    if token_present :
        if store_acct_to_disk :
            # Store token to disk
            print("Storing token to disk")
            IBMQ.save_account(token, qiskit_url, overwrite=True)
        else :
            # Use token only for the current session
            print("Using token for current session only")
            IBMQ.enable_account(token)
