# https://github.com/objectnf/d3ctf_easyQuantum/blob/3ce0a7b28ec349033965efb638e41669e263bc52/solve.py
import pyshark
import binascii
import qiskit
import pickle
from bitstring import BitArray


QUANLENG = 4


key = ""
cap = pyshark.FileCapture('cap2.pcapng')


def decrypt_msg(enckey: BitArray, msg: BitArray):
    res = BitArray()
    for i in range(msg.len):
        tmp = enckey[i] ^ msg[i]
        res.append("0b" + str(int(tmp)))
    return res


def recv_quantum(quantum_state: list):
    # Load state
    quantum = [qiskit.QuantumCircuit(1, 1) for _ in range(4)]
    real_part_a = quantum_state[0][0].real
    real_part_b = quantum_state[0][1].real
    if real_part_a == 1.0 and real_part_b == 0.0:
        continue
    elif real_part_a == 0.0 and real_part_b == 1.0:
        quantum[0].x(0)
    elif real_part_a > 0.7 and real_part_b > 0.7:
        quantum[0].h(0)
    else:
        quantum[0].x(0)
        quantum[0].h(0)
    real_part_a = quantum_state[1][0].real
    real_part_b = quantum_state[1][1].real
    if real_part_a == 1.0 and real_part_b == 0.0:
        continue
    elif real_part_a == 0.0 and real_part_b == 1.0:
        quantum[1].x(0)
    elif real_part_a > 0.7 and real_part_b > 0.7:
        quantum[1].h(0)
    else:
        quantum[1].x(0)
        quantum[1].h(0)
    real_part_a = quantum_state[2][0].real
    real_part_b = quantum_state[2][1].real
    if real_part_a == 1.0 and real_part_b == 0.0:
        continue
    elif real_part_a == 0.0 and real_part_b == 1.0:
        quantum[2].x(0)
    elif real_part_a > 0.7 and real_part_b > 0.7:
        quantum[2].h(0)
    else:
        quantum[2].x(0)
        quantum[2].h(0)
    real_part_a = quantum_state[3][0].real
    real_part_b = quantum_state[3][1].real
    if real_part_a == 1.0 and real_part_b == 0.0:
        continue
    elif real_part_a == 0.0 and real_part_b == 1.0:
        quantum[3].x(0)
    elif real_part_a > 0.7 and real_part_b > 0.7:
        quantum[3].h(0)
    else:
        quantum[3].x(0)
        quantum[3].h(0)
    return quantum


def measure(receiver_bases: list, quantum: list):
    if receiver_bases[0]:
        quantum[0].h(0)
        quantum[0].measure(0, 0)
    else:
        quantum[0].measure(0, 0)
    quantum[0].barrier()
    if receiver_bases[1]:
        quantum[1].h(0)
        quantum[1].measure(0, 0)
    else:
        quantum[1].measure(0, 0)
    quantum[1].barrier()
    if receiver_bases[2]:
        quantum[2].h(0)
        quantum[2].measure(0, 0)
    else:
        quantum[2].measure(0, 0)
    quantum[2].barrier()
    if receiver_bases[3]:
        quantum[3].h(0)
        quantum[3].measure(0, 0)
    else:
        quantum[3].measure(0, 0)
    quantum[3].barrier()
    # Execute
    backend = qiskit.Aer.get_backend("statevector_simulator")
    result = qiskit.execute(quantum, backend).result().get_counts()
    return result


def get_key(qubits: list, bases: list, compare_result: list):
    measure_result = measure(bases, qubits)
    if compare_result[0]:
        tmp_res = list(measure_result[0].keys())
        global key
        key += str(tmp_res[0])
    if compare_result[1]:
        tmp_res = list(measure_result[1].keys())
        global key
        key += str(tmp_res[0])
    if compare_result[2]:
        tmp_res = list(measure_result[2].keys())
        global key
        key += str(tmp_res[0])
    if compare_result[3]:
        tmp_res = list(measure_result[3].keys())
        global key
        key += str(tmp_res[0])


if __name__ == "__main__":
    key_len = pickle.loads(cap[0].data.data.binary_value)
    i = 1
    while i < 567:
        if int(cap[i+1].data.len) == 15:
            i += 2
            continue
        quantum_state = pickle.loads(cap[i].data.data.binary_value)
        quantum = recv_quantum(quantum_state)
        bob_bases = pickle.loads(cap[i+1].data.data.binary_value)
        alice_judge = pickle.loads(cap[i+2].data.data.binary_value)
        get_key(quantum, bob_bases, alice_judge)
        i += 3
    key = key[:key_len]
    msg = BitArray(pickle.loads(cap[567].data.data.binary_value))
    plaintext = decrypt_msg(BitArray("0b"+key), msg)
    print(plaintext.tobytes())
