# https://github.com/ethanhs/bloq/blob/a677bce262b8ce6a34e4063ac981e6b3474359da/python/qmc/backend.py
import asyncio
from pathlib import Path

from qiskit import Aer
from qiskit_ionq_provider import IonQProvider 

import random

from qmc.utils import to_thread

with open(Path(__file__).parent / "API_KEY") as key_file:
    key = key_file.read().strip()

provider = IonQProvider(token=key)

class Backend:
    """The base Backend class"""

def _run_ionq_simulator(qc):
    backend = provider.get_backend("ionq_simulator")
    job = backend.run(qc, shots=2)
    result = job.result().get_counts().keys()
    if len(result) == 1:
        return list(result)[0]
    else:
        return random.choice(list(result))

class IonQSimulatorBackend(Backend):
    async def schedule_execution(self, qc):
        result = await to_thread(_run_ionq_simulator, qc)
        return result

def _run_ionq_qpu(qc):
    backend = provider.get_backend("ionq_qpu")
    job = backend.run(qc, shots=1)
    result = job.result().get_counts()
    return list(result.keys())[0]

class IonQQPUBackend(Backend):
    async def schedule_execution(self, qc):
        result = await to_thread(_run_ionq_qpu, qc)
        return result