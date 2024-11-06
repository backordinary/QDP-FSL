# https://github.com/MSwenne/BEP/blob/f2848e3121e976540fb10171fdfbc6670dd28459/Code/site-packages/qiskit/providers/ibmq/ibmqaccounterror.py
# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Exception for errors raised by jobs.
"""

from qiskit import QiskitError


class IBMQAccountError(QiskitError):
    """Base class for errors raised by account management."""
    pass
