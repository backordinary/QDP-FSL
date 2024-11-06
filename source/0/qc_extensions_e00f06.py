# https://github.com/donn/djsat-eval/blob/829475d98a7a9ac57cc0b3771af536f93a4a6ce0/lib/qc_extensions.py
from qiskit import QuantumCircuit, Aer, IBMQ
from qiskit import execute, transpile

from qiskit.quantum_info import Statevector
from qiskit.providers import Backend, ibmq
from qiskit.tools.monitor import job_monitor
from qiskit.result import Result

from numpy import ndarray
import numpy as np


def get_ibm_backend(qubits_needed: int) -> Backend:
    provider = IBMQ.load_account()
    provider = IBMQ.get_provider(hub="ibm-q")
    backend = ibmq.least_busy(
        provider.backends(
            filters=lambda x: x.configuration().n_qubits >= (qubits_needed)
            and not x.configuration().simulator
            and x.status().operational == True
        )
    )
    print(f"Least busy backend: {backend}")
    return backend


def __simulate(circuit: QuantumCircuit, shots: int = 1024) -> Result:
    backend = Aer.get_backend("qasm_simulator")
    job = execute(circuit, backend, shots=shots)
    result = job.result()
    return result


def __run(
    self: QuantumCircuit, backend: Backend, transpiled=False, tp_kwargs=None
) -> Result:
    if tp_kwargs is None:
        tp_kwargs = {"optimization_level": 3}

    circ = self
    if not transpiled:
        circ = self.transpile(backend=backend, **tp_kwargs)
    job = backend.run(circ)
    job_monitor(job, interval=1)
    return job.result()


def __sv(self: QuantumCircuit) -> ndarray:
    a = Statevector.from_instruction(self).data.copy()
    a[np.abs(a) < np.finfo(float).eps] = 0
    return a


def __visualize(self: QuantumCircuit):
    return self.draw("latex", scale=2)


def __transpile(self: QuantumCircuit, **kwargs) -> QuantumCircuit:
    return transpile(circuits=self, **kwargs)


QuantumCircuit.simulate = __simulate
QuantumCircuit.sv = __sv
QuantumCircuit.visualize = __visualize
QuantumCircuit.transpile = __transpile
QuantumCircuit.run = __run


__all__ = ["get_ibm_backend"]
