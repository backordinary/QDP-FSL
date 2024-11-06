# https://github.com/qiskit-community/qiskit-qcgpu-provider/blob/5386d3c9110d7e71f4c97d756c21724f3f417498/qiskit_qcgpu_provider/simulatorerror.py
"""
Exception for errors raised by QCGPU simulators
"""

from qiskit import QiskitError


class QCGPUSimulatorError(QiskitError):
    """Base class for errors raised by simulators."""

    def __init__(self, *message):
        """Set the error message"""
        super().__init__(*message)
        self.message = ' '.join(message)

    def __str__(self):
        """Return the message"""
        return repr(self.message)
