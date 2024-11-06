# https://github.com/rum-yasuhiro/multi-tasking_transpiler/blob/8af865f169f51dbfb0bf9eb402e5d7ffd06ca038/test/conftest.py
"""
This module povides quantum circuit style qasm bench for pytest
"""
# module path is pointing to palloq/palloc
import os
import pathlib
import pytest
import logging

from qiskit import QuantumCircuit
from datetime import datetime
from qiskit.providers.models import BackendProperties
from qiskit.providers.models.backendproperties import Nduv, Gate

_log = logging.getLogger(__name__)


def load_qasm(size: str):
    """
    This function loads qasm files
    Arguments:
        size: (str) circuit size (small, medium, large
    """
    pos = os.path.abspath(os.path.dirname(__file__))
    test_files = str(pathlib.Path(pos).parent)
    QASM_BENCH = pathlib.Path(test_files + "/qasm_bench")
    if not QASM_BENCH.exists():
        raise Exception("Something wrong with path settings for test")

    if size == "small":
        small_path = QASM_BENCH.joinpath("small")
        return small_path
    elif size == "medium":
        medium_path = QASM_BENCH.joinpath("medium")
        return medium_path
    elif size == "large":
        large_path = QASM_BENCH.joinpath("large")
        return large_path
    else:
        raise ValueError("size must be small, medium or large")


def collect(path):
    # circuits = [qc.glob("*.qasm") for qc in path.iterdir()]
    circuits = [str(i) for qc in path.iterdir() 
                for i in qc.glob("[a-z]*.qasm")]
    return circuits

@pytest.fixture(scope="session")
def small_circuits() -> list:
    """
    This function returns a list of small quantum circuits
    """
    path = load_qasm("small")
    circuits = []
    for qc_qasm in collect(path):
        try:
            qc = QuantumCircuit.from_qasm_file(qc_qasm)
            circuits.append(qc)
        except Exception:
            _log.warning(f"Parsing Qasm Failed at {qc_qasm}")
            continue
    return circuits


@pytest.fixture(scope="session")
def medium_circuits() -> list:
    """
    This function returns a list of medium quantum circuits
    """
    path = load_qasm("medium")
    circuits = []
    for qc_qasm in collect(path):
        try:
            qc = QuantumCircuit.from_qasm_file(qc_qasm)
            circuits.append(qc)
        except Exception:
            _log.warning(f"Parsing Qasm Failed at {qc_qasm}")
            continue
    return circuits


@pytest.fixture(scope="session")
def large_circuits() -> list:
    """
    This function returns a list of large quantum circuits
    """
    path = load_qasm("large")
    circuits = []
    for qc_qasm in collect(path):
        try:
            qc = QuantumCircuit.from_qasm_file(qc_qasm)
            circuits.append(qc)
        except Exception:
            _log.warning(f"Parsing Qasm Failed at {qc_qasm}")
            continue
    return circuits


def make_noisy_qubit(t_1=50.0, t_2=50.0):
    """Create a qubit for BackendProperties"""
    calib_time = datetime(year=2019, month=2, day=1, hour=0, minute=0, second=0)
    return [Nduv(name="T1", date=calib_time, unit="µs", value=t_1),
            Nduv(name="T2", date=calib_time, unit="µs", value=t_2),
            Nduv(name="frequency", date=calib_time, unit="GHz", value=5.0),
            Nduv(name="readout_error", date=calib_time, unit="", value=0.01)]

@pytest.fixture(scope="session")
def fake_machine():
    """Create a 6 qubit machine to test crosstalk adaptive schedules"""
    calib_time = datetime(year=2019, month=2, day=1, hour=0, minute=0, second=0)
    qubit_list = []
    qubit_list.append(make_noisy_qubit())
    qubit_list.append(make_noisy_qubit())
    qubit_list.append(make_noisy_qubit())
    qubit_list.append(make_noisy_qubit())
    qubit_list.append(make_noisy_qubit())
    qubit_list.append(make_noisy_qubit())
    cx01 = [Nduv(date=calib_time, name='gate_error', unit='', value=0.05),
            Nduv(date=calib_time, name='gate_length', unit='ns', value=450.0)]
    cx12 = [Nduv(date=calib_time, name='gate_error', unit='', value=0.05),
            Nduv(date=calib_time, name='gate_length', unit='ns', value=451.0)]
    cx23 = [Nduv(date=calib_time, name='gate_error', unit='', value=0.05),
            Nduv(date=calib_time, name='gate_length', unit='ns', value=452.0)]
    cx34 = [Nduv(date=calib_time, name='gate_error', unit='', value=0.05),
            Nduv(date=calib_time, name='gate_length', unit='ns', value=453.0)]
    cx45 = [Nduv(date=calib_time, name='gate_error', unit='', value=0.05),
            Nduv(date=calib_time, name='gate_length', unit='ns', value=454.0)]
    gcx01 = Gate(name="CX0_1", gate="cx", parameters=cx01, qubits=[0, 1])
    gcx12 = Gate(name="CX1_2", gate="cx", parameters=cx12, qubits=[1, 2])
    gcx23 = Gate(name="CX2_3", gate="cx", parameters=cx23, qubits=[2, 3])
    gcx34 = Gate(name="CX3_4", gate="cx", parameters=cx34, qubits=[3, 4])
    gcx45 = Gate(name="CX4_5", gate="cx", parameters=cx45, qubits=[4, 5])
    u_1 = [Nduv(date=calib_time, name='gate_error', unit='', value=0.001),
           Nduv(date=calib_time, name='gate_length', unit='ns', value=100.0)]
    gu10 = Gate(name="u1_0", gate="u1", parameters=u_1, qubits=[0])
    gu11 = Gate(name="u1_1", gate="u1", parameters=u_1, qubits=[1])
    gu12 = Gate(name="u1_2", gate="u1", parameters=u_1, qubits=[2])
    gu13 = Gate(name="u1_3", gate="u1", parameters=u_1, qubits=[3])
    gu14 = Gate(name="u1_4", gate="u1", parameters=u_1, qubits=[4])
    gu15 = Gate(name="u1_4", gate="u1", parameters=u_1, qubits=[5])
    u_2 = [Nduv(date=calib_time, name='gate_error', unit='', value=0.001),
           Nduv(date=calib_time, name='gate_length', unit='ns', value=100.0)]
    gu20 = Gate(name="u2_0", gate="u2", parameters=u_2, qubits=[0])
    gu21 = Gate(name="u2_1", gate="u2", parameters=u_2, qubits=[1])
    gu22 = Gate(name="u2_2", gate="u2", parameters=u_2, qubits=[2])
    gu23 = Gate(name="u2_3", gate="u2", parameters=u_2, qubits=[3])
    gu24 = Gate(name="u2_4", gate="u2", parameters=u_2, qubits=[4])
    gu25 = Gate(name="u2_4", gate="u2", parameters=u_2, qubits=[5])
    u_3 = [Nduv(date=calib_time, name='gate_error', unit='', value=0.001),
           Nduv(date=calib_time, name='gate_length', unit='ns', value=100.0)]
    gu30 = Gate(name="u3_0", gate="u3", parameters=u_3, qubits=[0])
    gu31 = Gate(name="u3_1", gate="u3", parameters=u_3, qubits=[1])
    gu32 = Gate(name="u3_2", gate="u3", parameters=u_3, qubits=[2])
    gu33 = Gate(name="u3_3", gate="u3", parameters=u_3, qubits=[3])
    gu34 = Gate(name="u3_4", gate="u3", parameters=u_3, qubits=[4])
    gu35 = Gate(name="u3_5", gate="u3", parameters=u_3, qubits=[5])

    gate_list = [gcx01, gcx12, gcx23, gcx34, gcx45,
                 gu10, gu11, gu12, gu13, gu14, gu15,
                 gu20, gu21, gu22, gu23, gu24, gu25,
                 gu30, gu31, gu32, gu33, gu34, gu35]

    bprop = BackendProperties(last_update_date=calib_time, backend_name="test_backend",
                              qubits=qubit_list, backend_version="1.0.0", gates=gate_list,
                              general=[])
    return bprop
