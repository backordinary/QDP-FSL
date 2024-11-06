# https://github.com/iamtxena/quantum-channel-discrimination/blob/b1022103eb8126e409dd940a25940efcdb1c2943/qcd/backends/statevector_simulator.py
from . import DeviceBackend
from qiskit import Aer


class StateVectorSimulatorBackend(DeviceBackend):

    """ Representation of the Aer State Vector Simulator backend """

    def __init__(self) -> None:
        self._backend = Aer.get_backend('statevector_simulator')
        super().__init__(self._backend)
