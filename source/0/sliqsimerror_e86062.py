# https://github.com/NTU-ALComLab/SliQSim-Qiskit-Interface/blob/6c8b23050bce87b9851f224d661a57eb462d2642/qiskit_sliqsim_provider/sliqsimerror.py
# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Exception for errors raised by SliQSim simulator.
"""


from qiskit import QiskitError


class SliQSimError(QiskitError):
    """Class for errors raised by the SliQSim simulator."""

    def __init__(self, *message):
        """Set the error message."""
        super().__init__(*message)
        self.message = ' '.join(message)

    def __str__(self):
        """Return the message."""
        return repr(self.message)
