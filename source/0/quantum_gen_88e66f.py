# https://github.com/leopeltola/quantum-seed-generator/blob/1ae214317e3a08382073a8fb19babd426dce6816/app/quantum_gen.py
# PRNG with Pseudo Quantum Error
# 2021 Â© Aaron Campbell


from os import environ
import math
from qiskit import *
from qiskit.tools.monitor import job_monitor
from qiskit.tools.visualization import plot_histogram
from qiskit.providers.ibmq import least_busy


#IBM setup
if environ.get("FLASK_ENV") == "production":
    ibm_token = environ.get("IBM_TOKEN")
    IBMQ.enable_account(ibm_token)
else:
    IBMQ.load_account()


def generate_seeds():
    # select the quantum computer and update number_of_qbits to match
    provider = IBMQ.get_provider('ibm-q')
    qcomp = least_busy(provider.backends(
        filters=lambda x: x.configuration().n_qubits >= 3 
        and not x.configuration().simulator 
        and x.status().operational==True))
    number_of_qbits = len(qcomp.properties().qubits)

    # set up the quantum circuit
    qr = QuantumRegister(number_of_qbits)
    cr = ClassicalRegister(number_of_qbits)
    circuit = QuantumCircuit(qr,cr)

    # put the first qbit into a superposition
    circuit.h(qr[0])

    # define the other qbits values based on the first's
    for i in range(number_of_qbits-1):
        circuit.cx(qr[0],qr[i+1])
    
    # prepare to measure the state of the circuit
    circuit.measure(qr,cr)

    # run the job and produce a list of cuttoff percentages
    job = execute(circuit, backend=qcomp)
    job_monitor(job)
    results = job.result().get_counts(circuit)

    return [*results.values()]
