# https://github.com/timcp/StabRankSearcher/blob/6fdd9b0ce3b69459a713424a0a787edd10e6b0c5/stabranksearcher/stab_basis_provider/random.py
import numpy as np
from netsquid.qubits.stabtools import StabRepr
import netsquid.qubits.qubitapi as qapi
import qiskit
from stabranksearcher.basis import Basis
from stabranksearcher.stab_basis_provider.stab_basis_provider import StabBasisProvider


class RandomStabBasisProvider(StabBasisProvider):

    def __init__(self, number_of_qubits=1, stabrank=1):
        self._number_of_qubits = number_of_qubits
        self._stabrank = stabrank

    def get_next_basis(self):
        return RandomStabBasisProvider.get_random_stabilizer_state_basis(
                    number_of_qubits=self._number_of_qubits,
                    size=self._stabrank)

    @classmethod
    def get_random_stabilizer_state_basis(cls, number_of_qubits=1, size=1):
        qstates = set()
        while len(list(qstates)) != size:
            stabstate = cls.get_random_stabilizer_state(number_of_qubits=number_of_qubits)
            qstates.add(stabstate)
        return Basis(qstates=list(qstates))

    @classmethod
    def get_random_stabilizer_state(cls, number_of_qubits=1):

        # get random clifford
        cliff = qiskit.quantum_info.random_clifford(num_qubits=number_of_qubits)
        phases = cliff.stabilizer.phase
        check_matrix_x = cliff.stabilizer.X
        check_matrix_z = cliff.stabilizer.Z
        full_check_matrix = np.hstack((check_matrix_x, check_matrix_z))

        # convert to format that netsquid wants:
        # - check matrix with zeroes and ones
        # - phases are +1 or -1
        full_check_matrix = full_check_matrix.astype(int)
        phases = phases.astype(int)
        phases = [1 if phase == 1. else -1 for phase in phases]

        # make netsquid stabilizer repr
        srepr = StabRepr(check_matrix=full_check_matrix, phases=phases)
        qubits = qapi.create_qubits(num_qubits=number_of_qubits)
        qapi.assign_qstate(qubits, srepr)
        return qubits[0].qstate
