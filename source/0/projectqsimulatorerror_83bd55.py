# https://github.com/yaelbh/qiskit-projectq-provider/blob/7b87bd750e33945362c63445ef7b8aae6ffc0497/qiskit_addon_projectq/projectqsimulatorerror.py
# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Exception for errors raised by ProjectQ simulators.
"""

from qiskit import QISKitError


class ProjectQSimulatorError(QISKitError):
    """Class for errors raised by the ProjectQ simulators."""

    def __init__(self, *message):
        """Set the error message."""
        super().__init__(*message)
        self.message = ' '.join(message)

    def __str__(self):
        """Return the message."""
        return repr(self.message)
