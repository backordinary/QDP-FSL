# https://github.com/Kinshukg04/QuantCertificate/blob/8e48590ea6da1783c845b5e76e4195c72e450ffd/app/src/qcalgos.py
from qiskit import QuantumCircuit, execute, Aer
from qiskit.visualization import plot_histogram, plot_bloch_multivector
import numpy as np
from PIL import Image
import imagehash
from Crypto.Cipher import AES

def random_binary_array(size=256):
    return np.random.randint(2, size=size)

def encode_message(bits, bases):
    """Prepares qubits for given bits using bases transform
    """
    size = len(bases)

    message = []
    for i in range(size):
        qc = QuantumCircuit(1,1)
        if bases[i]==0: # Prepare qubit in Z-basis
            if bits[i]==0:
                pass
            else:
                qc.x(0)
        else:           # Prepare qubit in X-basis
            if bits[i]==0:
                qc.h(0)
            else:
                qc.x(0)
                qc.h(0)
        qc.barrier()
        message.append(qc)
    return message

def decode_message(message, bases):
    """Measures message of qubits using given bases
    """
    backend = Aer.get_backend('qasm_simulator')
    n = len(message)
    measurements = np.zeros(n, dtype=np.int)

    for q in range(n):
        if bases[q]==0: # Measuring in Z-basis
            message[q].measure(0,0)
        if bases[q]==1: # Measuring in X-basis
            message[q].h(0)
            message[q].measure(0,0)
        result = execute(message[q], backend, shots=1, memory=True).result()
        measured_bit = int(result.get_memory()[0])
        measurements[q] = measured_bit
    return measurements

def bit_intersection(sender_bases, receiver_bases, bits):
    final_bits = []
    n = len(sender_bases)
    for q in range(n):
        if sender_bases[q]==receiver_bases[q]:
            final_bits.append(bits[q])
    return final_bits

def sample_bits(bits, selection):
    sample = []
    for i in selection:
        i = np.mod(i, len(bits))
        sample.append(bits.pop(i))
    return sample

def create_quantum_shared_key(max_size=256):
    """Returns quantum shared key for 2p party

    Parameters
    -----------------------
    max_size: Maximum size of shared key
    """
    # Create random message (bits) and random bases to prepare qubits by sender
    sender_bits = random_binary_array(size=max_size)
    sender_bases = random_binary_array(size=max_size)

    # Send the message to recevier
    message = encode_message(sender_bits, sender_bases)

    # Decode message
    receiver_bases = random_binary_array(size=max_size)
    receiver_results = decode_message(message, receiver_bases)

    # Create keys
    sender_key = bit_intersection(sender_bases, receiver_bases, sender_bits)
    receiver_key = bit_intersection(sender_bases, receiver_bases, receiver_results)
    
    # Send sender and receiver sample over transmission medium
    sample_size = np.random.randint(len(sender_key))
    bit_selection = np.random.randint(len(sender_key), size=sample_size)
    sender_sample = sample_bits(sender_key, bit_selection)
    receiver_sample = sample_bits(receiver_key, bit_selection)

    assert (sender_sample==receiver_sample), "Somebody is trying to intercept the message"
    shared_key = sender_key # or receiver key as both must be same

    if len(shared_key)==0 or sender_key!=receiver_key:
        # Repeat once more
        # TODO: Remove huge stack call (extremely rare situation or impossible though)
        return create_quantum_shared_key(max_size=max_size)
    
    shared_key = (max_size - len(shared_key))*'0' + ''.join([str(s) for s in shared_key])
    key = ''
    for i in range(0, len(shared_key), 8): 
        temp_data = shared_key[i:i + 8] 
        decimal_data = int(temp_data, 2) 
        key = key + chr(decimal_data)

    return key

def hash_image(image, hash_size=256):
    """Returns binary hash numpy array of length hash size encoding image

    Parameters
    ------------------------
    image: Image object (PIL Image)
    hash_size: Length of output binary hash array
    """
    if hash_size%4!=0:
        raise Exception(f"Hash size {hash_size} must be even power of 2")

    hash = imagehash.whash(
        image=image,
        hash_size=int(np.sqrt(hash_size))
    )
    # Convert to binary array and pad zeros
    hash = bin(int(str(hash), 16)).replace("0b", "")
    hash = (hash_size-len(hash))*"0"+hash
    hash = np.array(list(map(int, hash)))

    assert(hash.shape==(hash_size,)), "Unexpected error in calculating message hash"
    hash = "".join([str(s) for s in hash])
    
    return hash

def encrypt(key, data, key_size=256):
    # Make key in suitable format
    cipher = AES.new(key.encode('utf8')[:32], AES.MODE_EAX)
    nonce = cipher.nonce
    ciphertext, tag = cipher.encrypt_and_digest(data.encode('utf8'))
    return nonce, ciphertext, tag

def decrypt(key, nonce, ciphertext, tag):
    cipher = AES.new(key.encode('utf8')[:32], AES.MODE_EAX, nonce=nonce)
    plaintext = cipher.decrypt(ciphertext)
    try:
        cipher.verify(tag)
        print("The message is authentic:", plaintext)
    except ValueError:
        print("Key incorrect or message corrupted")
    return plaintext


