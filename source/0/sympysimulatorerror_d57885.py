# https://github.com/yaelbh/qiskit-sympy-provider/blob/57d2f3e1e95fedd8e5a771084e2cb538b0e18401/qiskit_addon_sympy/sympysimulatorerror.py
# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Exception for errors raised by Sympy simulators.
"""

from qiskit import QISKitError


class SympySimulatorError(QISKitError):
    """Class for errors raised by the Sympy simulators."""

    def __init__(self, *message):
        """Set the error message."""
        super().__init__(*message)
        self.message = ' '.join(message)

    def __str__(self):
        """Return the message."""
        return repr(self.message)
