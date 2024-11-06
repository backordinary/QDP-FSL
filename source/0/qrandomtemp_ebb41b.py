# https://github.com/iQuHack-2023-Qoalas/qoalasFlask/blob/d00988940c9a8331a46e87596ff73cb14b3260aa/qrandomtemp.py
"""
This module utilizes quantum computers 
to produce truly random numbers
"""
from qiskit import IBMQ, transpile, execute
from qiskit import IBMQ, Aer
from qiskit.tools.monitor import job_monitor
from generators import NormalDistribution,UniformDistribution,PorterThomasDistribution

import yaml


def generateNumbers():
    IBMQ.save_account('001725cf4ad0eafc1d267990a28fafc71dc62783be9ba638874f69dd2139964f2fe7209c23732257dc5f69ab26421772e46aae5c6dea61afac980b1632912116', overwrite=True)
    IBMQ.load_account() # Load account from disk
    IBM_provider = IBMQ.get_provider(hub='ibm-q-community')
    ibm_nairobi = IBM_provider.backends('ibm_nairobi')[0]
    ibm_oslo = IBM_provider.backends('ibm_oslo')[0]
    # ibm_sim_online = IBM_provider.backends('ibmq_qasm_simulator')[0]
    ibm_sim_local = Aer.get_backend('qasm_simulator')



    shots = 10
    numb_qubits = 7
    distribution_type = ['normal', 'uniform', 'lognormal', 'porterthomas'][0]
    numb_of_random = 10
    backend = [ibm_nairobi, ibm_oslo, ibm_sim_local][2]
    filename = 'seeds.yml'

    for r in range(numb_of_random):
        if distribution_type == 'normal':
            circuit = NormalDistribution(num_qubits = numb_qubits)
        elif distribution_type == 'uniform':
            circuit = UniformDistribution(num_qubits = numb_qubits)
        elif distribution_type == 'lognormal':
            circuit = LogNormalDistribution(num_qubits = numb_qubits)
        elif distribution_type == 'porterthomas':
            circuit = PorterThomasDistribution(num_qubits = numb_qubits)

        circuit.measure_all()
        circuit = transpile(circuit, backend=backend)
        job = execute(circuit, ibm_sim_local, shots=shots)
        job_monitor(job)
        counts = job.result().get_counts()

        rnd_number = 0
        numb_shot = 1
        for k in counts.keys():
            while counts[k] > 0:
                rnd_number += int(k, 2) / (2**(numb_qubits*numb_shot))
                numb_shot += 1
                counts[k] -= 1

        # read seed data from yml file
        with open("seeds.yml") as file:
            data = yaml.safe_load(file)
            file.close()
        
        data[0]["seeds"].append(rnd_number)

        # update yml file
        with open("seeds.yml", 'w') as file:
            yaml.dump(data, file)
            file.close()