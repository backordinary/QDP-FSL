# https://github.com/MrLorent/JOY-DIV-AI/blob/ad96115a79f402fae0351de4534b4075ba80966b/backend/models.py
# MODELS

# Importing standard Qiskit libraries and configuring account
from qiskit import execute, Aer, IBMQ
from qiskit_aer.noise import NoiseModel
from qiskit.providers.ibmq import *
from qiskit.tools.monitor import job_monitor

def compute_circuits(circuits):
    # Loading your IBM Q account(s)
    print("\nInitialising connection to backend...")
    IBMQ.save_account('0ebf515971d26d25134acac3a0d10e36ddf8713377db17e66733b704c73b29b4fbe4fdaf0b55f51afd7e452956a0d4cae700dc5d0c9c286fd33a3dda610971dd', overwrite=True)
    IBMQ.load_account()
    provider = IBMQ.get_provider(hub = 'ibm-q')
    backend = provider.get_backend('ibm_oslo')
    print("Connected to", backend,"\n")

    # Get the noise from quantic
    noise_model = NoiseModel.from_backend(backend)
    basis_gates = noise_model.basis_gates

    # Run our circuit on a quantum computer simulator, using the pre. Monitor the execution of the job in the queue
    print("\nSend circuits to", backend,"...")
    job =   execute(
                circuits,
                Aer.get_backend('qasm_simulator'),
                basis_gates=basis_gates,
                noise_model=noise_model,
                shots=1024
            )
    job_monitor(job, interval = 2)
    print(backend, "response received.\n")

    # Get the results from the computation
    results = job.result()
    answer = results.get_counts()

    return answer