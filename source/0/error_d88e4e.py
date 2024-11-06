# https://github.com/MSwenne/BEP/blob/f2848e3121e976540fb10171fdfbc6670dd28459/Code/site-packages/qiskit/tools/visualization/_error.py
# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Exception for errors raised by visualization module."""

from qiskit import QiskitError


class VisualizationError(QiskitError):
    """For visualization specific errors."""
    pass
