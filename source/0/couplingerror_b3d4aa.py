# https://github.com/MSwenne/BEP/blob/f2848e3121e976540fb10171fdfbc6670dd28459/Code/site-packages/qiskit/mapper/_couplingerror.py
# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Exception for errors raised by the CouplingMap object.
"""
from qiskit import QiskitError


class CouplingError(QiskitError):
    """Base class for errors raised by the coupling graph object."""

    def __init__(self, *msg):
        """Set the error message."""
        super().__init__(*msg)
        self.msg = ' '.join(msg)

    def __str__(self):
        """Return the message."""
        return repr(self.msg)
