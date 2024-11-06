# https://github.com/j0807s/ensembled_quantum_computing/blob/aa340cc17b7ab65b61393cd080b9ce96ecba40b3/main.py
import os
import sys
import qiskit
import numpy as np
import ray
import time
from qcirc_actor import RemoteWorker
from VQE import VQE
import pathlib
from QAOA import QAOA

ray.init()
path =str(pathlib.Path(__file__).parent.absolute())

if __name__ == "__main__":
    print("-"*20+"\nMain.py starting...\n"+"-"*20)
    k=4
    # ========================================================================================
    # Generate a list of backends using the qiskit provider.backends call
    # This can be a combination of providers, so long as they are independantly addressable
    # ========================================================================================
    #backends = [None]
    qiskit.IBMQ.load_account()
    provider = qiskit.IBMQ.get_provider(hub='ibm-q', group='open', project='main')
    #backends = provider.backends(simulator=False)
    all_backends = provider.backends(simulator=False)
    
    backends_list = ['ibmq_quito', 'ibmq_lima', 'ibm_oslo', 'ibm_nairobi', \
                'ibmq_belem', 'ibmq_manila']

    backends = [c for c in all_backends if (c.name() in backends_list)]
    
    # ========================================================================================
    # VQE and QAOA are two classes described in this package. They are examples of how to build
    # a variational quantum algorithm object, and then implement it onto EQC.
    # They inherit from the TrainableCircuit class.
    # ========================================================================================
    vqe = VQE(k)
    vqe.initialize()
    vqe.set_backend(backends)
    vqe.train(learning_rate=.1)

