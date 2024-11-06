# https://github.com/iamtxena/quantum-channel-discrimination/blob/618f34f473a9ae37090df1059de9bab9f97aa5cd/qcd/backends/simulator.py
from . import DeviceBackend
from qiskit import Aer


class SimulatorBackend(DeviceBackend):

    """ Representation of the Aer Simulator backend """

    def __init__(self) -> None:
        self._backend = Aer.get_backend('qasm_simulator')
        super().__init__(self._backend)
