# https://github.com/cda-tum/ddsim/blob/236821f556492dc3cce2c6d528626a1f08423a8a/mqt/ddsim/error.py
"""
Exception for errors raised by DDSIM simulator.
"""

from qiskit import QiskitError


class DDSIMError(QiskitError):
    """Class for errors raised by the DDSIM simulator."""

    def __init__(self, *message):
        """Set the error message."""
        super().__init__(*message)
