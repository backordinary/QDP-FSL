# https://github.com/niefermar/CuanticaProgramacion/blob/cf066149b4bd769673e83fd774792e9965e5dbc0/qiskit/dagcircuit/_dagcircuiterror.py
# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Exception for errors raised by the DAGCircuit object.
"""
from qiskit import QISKitError


class DAGCircuitError(QISKitError):
    """Base class for errors raised by the DAGCircuit object."""

    def __init__(self, *msg):
        """Set the error message."""
        super().__init__(*msg)
        self.msg = ' '.join(msg)

    def __str__(self):
        """Return the message."""
        return repr(self.msg)
