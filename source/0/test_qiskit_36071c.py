# https://github.com/qcware/qcware_transpile/blob/6822b68d02dac7ca32643e107a4f8eb974cc0b02/tests/dialects/test_qiskit.py
import traceback
import warnings
import sys

# this trick is here as well to try and pin down where qiskit is throwing
# more errors on import
def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    log = sys.stdout  # file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))


# warnings.showwarning = warn_with_traceback


from ..strategies.qiskit import circuits
from qcware_transpile.dialects.qiskit import (
    native_to_ir,
    ir_to_native,
    native_circuits_are_equivalent,
    normalized_instructions,
)
from hypothesis import given, note, settings
import qiskit
import numpy as np


@given(circuits(min_qubits=1, max_qubits=4, min_length=1, max_length=3))
@settings(deadline=None)
def test_conversion(qc):
    note(qc.draw())
    note(qc.data)
    c = native_to_ir(qc)
    note(c)
    qc2 = ir_to_native(c)
    note(qc2.draw())
    note(qc2.data)
    instruction_pairs = zip(normalized_instructions(qc), normalized_instructions(qc2))
    for x in instruction_pairs:
        assert x[0] == x[1]
    assert native_circuits_are_equivalent(qc, qc2)
