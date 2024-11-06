# https://github.com/buehlefs/qc2-exercise/blob/b4e46f17c9de62c780ee3ae89fdfcead67f20960/backend/backend.py
from json import dumps

from qiskit import QuantumCircuit
from qiskit.providers import BackendV2 as Backend
from qiskit.providers import Options
from qiskit.transpiler import Target
from requests import request

from .job import QHAnaJob
from .util import text_to_data_url


class QHAnaBackend(Backend):
    """A qiskit backend for executing jobs with a QHAna plugin.
    
    Always call the ``run`` function of this backend directly!

    Note: Do not use this backend with the ``execute`` function
    as this function will attempt to transpile the circuit for 
    this backend which will fail!
    """

    def __init__(self, **kwargs):
        """Initialize the backend with a dummy target."""
        super().__init__(name="QHAna", **kwargs)

        self._target = Target("QHAna")

    @property
    def target(self):
        return self._target

    @property
    def max_circuits(self):
        return 1

    @classmethod
    def _default_options(cls):
        return Options(shots=1024)

    def run(self, circuits, **kwargs) -> QHAnaJob:
        shots = kwargs.get("shots", self.options.shots)

        if not isinstance(circuits, QuantumCircuit):
            raise TypeError("Unsupported input type!")

        # TODO: implement run method
