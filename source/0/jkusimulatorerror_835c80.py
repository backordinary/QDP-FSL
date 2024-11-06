# https://github.com/qiskit-community/qiskit-jku-provider/blob/0fb20bb8b0bf7a059f8c727641f1ff5625c7717a/qiskit_jku_provider/jkusimulatorerror.py
# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Exception for errors raised by JKU simulator.
"""


from qiskit import QiskitError


class JKUSimulatorError(QiskitError):
    """Class for errors raised by the JKU simulator."""

    def __init__(self, *message):
        """Set the error message."""
        super().__init__(*message)
        self.message = ' '.join(message)

    def __str__(self):
        """Return the message."""
        return repr(self.message)
