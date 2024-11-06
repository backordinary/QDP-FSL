# https://github.com/goldsmdn/Pipecleaning_test/blob/3109ddf0c231596ea03a4a9b1014cc05f09adee5/test_circuits.py
import pytest
from circuits import SteaneCodeLogicalQubit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer

TEST_X_QUBIT = 4
SHOTS = 100     #Number of shots to run    
SIMULATOR = Aer.get_backend('qasm_simulator')

def test_parity_validation():
    """test that a random errors get fixed"""
    parity_check_matrix =  [[0,0,0,1,1,1,1],
                            [0,1,1,0,0,1,1],
                            [1,0,1,0,1,0,1]]
    qubit = SteaneCodeLogicalQubit(1, parity_check_matrix, True)
    qubit.set_up_logical_zero(0)
    qubit.force_X_error(0,0)   #force X error for testing
    qubit.set_up_ancilla(0)
    qubit.decode(0)
    result = execute(qubit, SIMULATOR, shots=SHOTS).result()
    length = len(result.get_counts(qubit))
    assert length == 1
    qubit = SteaneCodeLogicalQubit(1, parity_check_matrix, True)
    qubit.set_up_logical_zero(0)
    qubit.force_X_error(1,0)   #force X error for testing
    qubit.set_up_ancilla(0)
    qubit.decode(0)
    result = execute(qubit, SIMULATOR, shots=SHOTS).result()
    length = len(result.get_counts(qubit))
    assert length == 1
    qubit = SteaneCodeLogicalQubit(1, parity_check_matrix, True)
    qubit.set_up_logical_zero(0)
    qubit.force_X_error(2,0)   #force X error for testing
    qubit.set_up_ancilla(0)
    qubit.decode(0)
    result = execute(qubit, SIMULATOR, shots=SHOTS).result()
    length = len(result.get_counts(qubit))
    assert length == 1
    qubit = SteaneCodeLogicalQubit(1, parity_check_matrix, True)
    qubit.set_up_logical_zero(0)
    qubit.force_X_error(3,0)   #force X error for testing
    qubit.set_up_ancilla(0)
    qubit.decode(0)
    result = execute(qubit, SIMULATOR, shots=SHOTS).result()
    length = len(result.get_counts(qubit))
    assert length == 1
    qubit = SteaneCodeLogicalQubit(1, parity_check_matrix, True)
    qubit.set_up_logical_zero(0)
    qubit.force_X_error(4,0)   #force X error for testing
    qubit.set_up_ancilla(0)
    qubit.decode(0)
    result = execute(qubit, SIMULATOR, shots=SHOTS).result()
    length = len(result.get_counts(qubit))
    assert length == 1
    qubit = SteaneCodeLogicalQubit(1, parity_check_matrix, True)
    qubit.set_up_logical_zero(0)
    qubit.force_X_error(5,0)   #force X error for testing
    qubit.set_up_ancilla(0)
    qubit.decode(0)
    result = execute(qubit, SIMULATOR, shots=SHOTS).result()
    length = len(result.get_counts(qubit))
    assert length == 1
    

       