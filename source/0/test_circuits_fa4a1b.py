# https://github.com/qcware/qiskit_qcware/blob/d89cb18f6c74b66058768fb7bc8ca5a122a3e8b1/tests/test_circuits.py
from qcware_transpile.dialects import qiskit as qiskit_dialect
from qcware_transpile.translations.qiskit.to_quasar import ( # type: ignore
    translation_set, native_is_translatable)  # type: ignore
from .strategies.qiskit import gates, circuits  # type: ignore
from qiskit_qcware import QcwareProvider
import quasar
import qiskit
from qiskit.providers.aer import AerSimulator
from hypothesis import given, settings, note, assume
import numpy

# we exclude reset gates because quasar doesn't support them, and we exclude
# measure gates because they interfere with statevector comparisons,
# which we use here to ensure circuit equality
ts = translation_set()
translatable_gatedefs = [
    x for x in qiskit_dialect.dialect().gate_defs if x.name not in {'measure', 'reset'}
]
translatable_circuits = circuits(1, 3, 1, 4,
                                 gates(gate_list=translatable_gatedefs))


def qcware_probability_vector(circuit: quasar.Circuit):
    backend = QcwareProvider().get_backend('local_statevector')
    sv = qiskit.execute(circuit, backend).result().data()['statevector']
    return abs(sv)


def aer_probability_vector(circuit: qiskit.QuantumCircuit):
    backend = AerSimulator(method="statevector")
    c = circuit.copy()
    c.save_state("final_statevector")
    sv = qiskit.execute(c, backend).result().data()['final_statevector']
    return abs(sv)


@given(translatable_circuits)
@settings(deadline=None)
def test_qiskit_qcware(qiskit_circuit):
    assume(native_is_translatable(qiskit_circuit))
    note(qiskit_circuit.draw())
    pv_qcware = qcware_probability_vector(qiskit_circuit)
    pv_aer = aer_probability_vector(qiskit_circuit)
    assert (numpy.allclose(pv_aer, pv_qcware, atol=1e-5))
