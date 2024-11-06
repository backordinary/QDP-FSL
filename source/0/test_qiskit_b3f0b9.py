# https://github.com/quantum-experiments/qbob/blob/6b33ea48a03e1e194dc87878b1d88395e560dff0/tests/test_qiskit.py
#!/usr/bin/env python

"""Tests for `qbob` integration with `qiskit`."""

import pytest
from typing import List

from qbob import qbob
from qbob.intrinsics import *
from qbob.types import *

from qiskit import *
from qbob.qiskit import *

def test_measure_entangled_state(measure_entangled_state):
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    
    my_qbob = qbob_from_qiskit(qc, "MeasureEntangledState")

    qsharp_code = my_qbob.build()
    print(qsharp_code)
    assert measure_entangled_state == qsharp_code
