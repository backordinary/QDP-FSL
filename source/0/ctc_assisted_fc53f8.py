# https://github.com/Baccios/CTC_iterator/blob/5b47d56702373edd9d928118f721475f676fbd0b/ctc/gates/ctc_assisted.py
"""
This module implements a Deutschian CTC gate
"""
from qiskit import transpile, Aer
from qiskit.circuit import Gate

import matplotlib.pyplot as plt


class CTCGate(Gate):
    """
    Deutschian CTC assisted gate.
    The first half qubits represent the CTC state, the second half the time dependent system.
    """

    def __init__(self, num_qubits, method="nbp", label=None):
        """

        :param num_qubits: The size (in qubits) of the gate. It must be even.
                 the first half represents the CTC
                 and the second half represents the Chronology Respecting (CR) system.
        :type num_qubits: int
        :param method: the recipe used to build the ctc assisted gate. It defaults to "nbp".
                   Possible values are:
                    <ol>
                        <li>"nbp": use the algorithm in
                            <a href="https://arxiv.org/abs/1901.00379">this article</a> (default value)
                        </li>
                        <li>"brun": use the algorithm in
                            <a href="https://arxiv.org/abs/0811.1209">this article</a>
                        </li>
                        <li>"brun_fig2": use the algorithm in
                            <a href="https://arxiv.org/abs/0811.1209">this article</a> in the variant for Fig.2.
                            (Note that this case is only limited to 2 bits)
                        </li>
                    </ol>
        :type method: str
        :param label: the label to use for the gate
        :type label: str
        """

        if num_qubits <= 0 or (num_qubits % 2) != 0:
            raise ValueError("num_qubits must be even.")

        self._num_qubits = num_qubits
        self._method = method

        super().__init__('CTC gate', num_qubits, [], label)

    def _define(self):
        """
        define gate behavior
        """
        from ctc.block_generator import get_ctc_assisted_circuit

        ctc_circuit = get_ctc_assisted_circuit(int(self._num_qubits / 2), method=self._method)

        # ctc_circuit.draw(output="mpl")  # DEBUG
        # plt.show()

        self.definition = ctc_circuit
