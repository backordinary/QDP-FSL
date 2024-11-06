# https://github.com/seunomonije/quantum-programming-api/blob/b2d45cdbf13b8e4d3917d9bea6317898da71aa33/qapi/protocols/cryptography/key_distribution/KeyGenerator.py
from qiskit import QuantumCircuit, execute, Aer
from qiskit.visualization import plot_histogram, plot_bloch_multivector
from numpy.random import randint
import numpy as np
import re

from qapi.protocols.cryptography.key_distribution.exceptions import InvalidBitstringError, InvalidQuantumKeyError

"""
  First iteration of key generation class. 
  Use:
    generator = KeyGenerator()
    generator.generate()
    generator.print_keys()

  Improvements:
    - Offload random number generation to C++
    - Figure out how to package/send encoded message outside of the class. Pickle? Would that work?
    - Test this out on the blockchain and see if we can get working results. 
    - Finish backend setup so that we're using the API to help itself.
"""
class KeyGenerator:
  def __init__(self, bit_string=None, bit_string_length=200):
    """
      Sets the bit string for key generation, or the desired length of a randomly 
      generated bitstring.

      Parameters:
        bit_string -- a given bitstring to base the key off of
        bit_string_length -- length of a desired random bitstring (Default 200)
    """
    if bit_string:
      # Check for invalid string
      if not re.match(r'^[0-1]{1,}$', bit_string):
        raise InvalidBitstringError('Your bit string input contains a non 0 or 1 character')
      
      if len(bit_string <= 100):
        raise InvalidBitstringError('Your bit string length input is too small. Must be more than 100.')

      self.bit_string = bit_string
      self.bit_string_length = len(bit_string)
    else:
      if bit_string_length <= 100:
        raise InvalidBitstringError('Your bit string length input is too small. Must be more than 100.')

      self.bit_string_length = bit_string_length
      self.bit_string = self._generate_random_bit_string()

  def generate(self):
    keys = self.build_keys()
    self.test_if_key_samples_match(*keys)
    self.keys = keys

  def build_keys(self):
    a_bits = self._generate_message()
    a_bases = self._generate_bases()

    message = self._encode_message_into_qubits(a_bits, a_bases)

    b_bases = self._generate_bases()
    b_measurements = self._measure_message(message, b_bases)

    a_key = self._remove_garbage(a_bases, b_bases, a_bits)
    b_key = self._remove_garbage(a_bases, b_bases, b_measurements)

    return (a_key, b_key)

  def test_if_key_samples_match(self, a_key, b_key, sample_size=15):
    bit_selection = randint(self.bit_string_length, size=sample_size)
    a_sample = self._sample_and_remove_bits(a_key, bit_selection)
    b_sample = self._sample_and_remove_bits(b_key, bit_selection)

    if a_sample == b_sample:
      return
    else:
      raise InvalidQuantumKeyError('The quantum generated keys failed the matching test')
  
  def print_keys(self):
    print(f'Key 1: {self.keys[0]}')
    print(f'Key 1 length: {len(self.keys[0])}')
    print('-----')
    print(f'Key 2: {self.keys[1]}')
    print(f'Key 2 length: {len(self.keys[1])}')

  def _generate_random_bit_string(self):
    bit_string = ""
    for _ in range(self.bit_string_length):
      random_bit_selection = randint(0, 2)
      bit_string += str(random_bit_selection)
    
    return bit_string

  def _generate_message(self):
    # Currently generates a random set of bits (0 or 1)
    return randint(2, size=self.bit_string_length)

  def _generate_bases(self):
    # Currently generates a random set of "bases", 0 or 1, which
    # stand for Z or X basis, respectively
    return randint(2, size=self.bit_string_length)

  def _encode_message_into_qubits(self, bits, measurement_bases):
    """
      Encodes the bit string message into qubits, which should be sent to the
      desired party to be measured over the desired channel.
    """
    message = []
    for i in range(self.bit_string_length):
      circuit = QuantumCircuit(1, 1)
      if measurement_bases[i] == 0:
        # Prepare qubit in Z-basis
        if bits[i] != 0:
          circuit.x(0)
      else:
        # Prepare qubit in X-basis
        if bits[i] == 0:
          circuit.h(0)
        else:
          circuit.x(0)
          circuit.h(0)
      circuit.barrier()
      message.append(circuit)
    return message
  
  def _measure_message(self, message, measurement_bases):
    """
      Handles measurement of the a provided qubit message given the provided bases.
      This should be used by the party receiving the encoded qubit message.
    """
    backend = Aer.get_backend('qasm_simulator')
    measurements = []
    for i in range(self.bit_string_length):
      if measurement_bases[i] == 0:
        message[i].measure(0,0)
      if measurement_bases[i] == 1:
        message[i].h(0)
        message[i].measure(0,0)
      
      result = execute(message[i], backend, shots=1, memory=True).result()
      measured_bit = int(result.get_memory()[0])
      measurements.append(measured_bit)
    return measurements

  def _remove_garbage(self, person_a_bases, person_b_bases, bits):
    """
      Removes any measured bits where the measurments occured in 
      differing basis.
    """
    good_bits = []
    for qubit in range(self.bit_string_length):
      if person_a_bases[qubit] == person_b_bases[qubit]:
        # If both used the same basis, add to the list of good bits
        good_bits.append(bits[qubit])
    return good_bits

  def _sample_and_remove_bits(self, bits, selection):
    """
      Samples a provided number of bits in a given key and returns
      them as a list
    """
    sample = []
    for index in selection:
      index = np.mod(index, len(bits))
      popped_element = bits.pop(index)
      sample.append(popped_element)
      # We're removing these bits from the array since we're broadcasting them publicly
    return sample
