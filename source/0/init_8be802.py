# https://github.com/qcware/qiskit_qcware/blob/975bb6e1d0f9a5b306f1d781b45c933dfdae14ff/tests/strategies/qiskit/__init__.py
import math
from hypothesis.strategies import (lists, integers, composite, sampled_from,
                                   floats)
import qiskit  # type: ignore
from qcware_transpile.dialects.qiskit.qiskit_dialect import (dialect,
                                                             name_to_class)


@composite
def gates(draw, gate_list=sorted(dialect().gate_defs)):
    gate_def = draw(sampled_from(gate_list))
    gate_class = name_to_class()[gate_def.name]
    kwargs = {}
    angles = floats(min_value=0, max_value=2 * math.pi)
    for p in gate_def.parameter_names:
        value = draw(angles)
        kwargs[p] = value
    try:
        result = gate_class(**kwargs)
    except Exception as e:
        print(
            f"Exception creating gate {gate_def.name}, class {gate_class.__name__} with parms {kwargs}"
        )
        raise e
    return result


@composite
def qreg_sizes(draw, num_qubits):
    """
    Make a list of quantum register sizes, the sizes of which sum to num_qubits
    """
    result = []
    while num_qubits > 0:
        i = draw(integers(min_value=1, max_value=num_qubits))
        result.append(i)
        num_qubits = num_qubits - i
    return result


@composite
def circuits(draw,
             min_qubits,
             max_qubits,
             min_length,
             max_length,
             gates=gates()):
    length = draw(integers(min_value=min_length, max_value=max_length))
    num_qubits = draw(integers(min_value=min_qubits, max_value=max_qubits))
    num_clbits = draw(integers(min_value=min_qubits, max_value=max_qubits))
    qregs = [qiskit.QuantumRegister(i) for i in draw(qreg_sizes(num_qubits))]

    circuit_gates = draw(
        lists(gates, min_size=length, max_size=length).filter(
            lambda x: all([y.num_qubits <= num_qubits for y in x])))

    if num_clbits > 0:
        cr = qiskit.ClassicalRegister(num_clbits)
        result = qiskit.QuantumCircuit(*qregs, cr)
    else:
        result = qiskit.QuantumCircuit(*qregs)
    for gate in circuit_gates:
        qubits = draw(
            lists(integers(min_value=0, max_value=num_qubits - 1),
                  min_size=gate.num_qubits,
                  max_size=gate.num_qubits,
                  unique=True))
        # manually muck about with measure
        if gate.name == 'measure':
            this_clbit = draw(integers(min_value=0, max_value=num_clbits - 1))
            result.append(gate, tuple(qubits), tuple([this_clbit]))
        else:
            result.append(gate, tuple(qubits))
    return result


# circuits(1,3,1,5).example()
