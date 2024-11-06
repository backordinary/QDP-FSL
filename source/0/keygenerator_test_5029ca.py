# https://github.com/seunomonije/quantum-programming-api/blob/b2d45cdbf13b8e4d3917d9bea6317898da71aa33/tests/protocols/cryptography/key_distribution/KeyGenerator_test.py
from qiskit import QuantumCircuit, execute, Aer
from qiskit.visualization import plot_histogram, plot_bloch_multivector
from numpy.random import randint
import numpy as np
import re

import os
import sys

from qapi.protocols.cryptography.key_distribution.KeyGenerator import KeyGenerator
from qapi.protocols.cryptography.key_distribution.exceptions import InvalidBitstringError, InvalidQuantumKeyError

# Setting a seed for testing purposes
np.random.seed(seed=0)

def create_and_run_generator(bit_string=None, bit_string_length=200):
  if bit_string:
    generator = KeyGenerator(bit_string=bit_string)
    generator.generate()
  else:
    generator = KeyGenerator(bit_string_length=bit_string_length)
    generator.generate()
  
  return generator
    

def test_invalid_bit_string():
  try:
    create_and_run_generator(bit_string_length=50)
    assert False
  except InvalidBitstringError:
    assert True

def test_non_binary_short_string():
  try:
    create_and_run_generator(bit_string='ABCDEFGHGF')
    assert False
  except InvalidBitstringError:
    assert True

def test_non_binary_long_string():
  try:
    create_and_run_generator(bit_string='ABCDEFGHGFABCDEFGHGFABCDEFGHGF\
    ABCDEFGHGFABCDEFGHGFABCDEFGHGFABCDEFGHGFABCDEFGHGFABCDEFGHGFABCDEFGHGFABCDEFGHGF')
    assert False
  except InvalidBitstringError:
    assert True

def test_successful_key_generation():
  generator = create_and_run_generator()
  expected_key = [1, 0, 1, 0, 1, 1, 0, 0, 0, 1, \
    1, 0, 0, 1, 0, 1, 1, 0, 0, 0, \
    0, 0, 0, 1, 1, 1, 0, 1, 1, 1, \
    0, 1, 1, 1, 1, 1, 1, 0, 1, 1, \
    1, 0, 0, 1, 1, 0, 1, 0, 0, 0, \
    1, 1, 1, 0, 1, 1, 1, 1, 1, 0, \
    1, 1, 1, 1, 1, 0, 0, 1, 0, 0, \
    1, 1, 0, 0, 0, 1, 1, 0, 1, 1, \
    1, 0, 1, 1, 1, 0]

  assert generator.keys[0] == expected_key and len(generator.keys[0]) == len(expected_key)
  assert generator.keys[1] == expected_key and len(generator.keys[1]) == len(expected_key)

  