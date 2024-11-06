# https://github.com/lbadams2/quantum_pattern_matching/blob/59936d7ee43620224a14e8470a29afefcf2f6b3e/src/oracles.py
'''
oracles.py

Liam Adams and Alec Landow

Creates oracles for each letter of the alphabet

References
    https://www.geeksforgeeks.org/iterate-over-characters-of-a-string-in-python/

'''

from qiskit import quantum_info
import numpy as np

# alphabet = ('a', 'b', 'c', 'd', 'e')
alphabet = ('0', '1')


def generate_many_oracles(s, input_string, pattern_length, debug, input_alphabet):
    global alphabet
    if input_alphabet:
        alphabet = input_alphabet

    oracles = dict()

    for pattern_char in alphabet:
        oracles[pattern_char] = list()
        N = len(input_string)
        M = pattern_length

        for i in range( N - M + 1 ):
            pattern_char_oracle_matrix = np.identity( int( 2**s ) )
            # print(f'{input_string[i]} =? {pattern_char}')
            if input_string[i] == pattern_char:
                pattern_char_oracle_matrix[i, i] = -1

                if debug:
                    print(f"Oracle for '{pattern_char}'[{i}]")
                    print(pattern_char_oracle_matrix)
                    print()
                oracles[pattern_char].append( quantum_info.Operator( pattern_char_oracle_matrix ) )

    oracles['*'] = list()
    wildcard_oracle_matrix = np.identity( int( 2**s ) )

    for i in range( N - M + 1 ):
        pattern_char_oracle_matrix[i, i] = -1

    oracles['*'].append( quantum_info.Operator( wildcard_oracle_matrix ) )

    return oracles

def generate_oracles_single(s, input_string, pattern_length, debug, input_alphabet):
    global alphabet
    if input_alphabet:
        alphabet = input_alphabet

    oracles = dict()
    for pattern_char in alphabet:

        pattern_char_oracle_matrix = np.identity( int( 2**s ) )
        # TODO: Imitate tensor products
        N = len(input_string)
        M = pattern_length
        # for i in range( 2**s ):
        for i in range( N - M + 1 ):
            # print(f'{input_string[i]} =? {pattern_char}')
            if input_string[i] == pattern_char:
                pattern_char_oracle_matrix[i, i] = -1

        if debug:
            print(f"Oracle for '{pattern_char}'")
            print(pattern_char_oracle_matrix)
            print()
        oracles[pattern_char] = quantum_info.Operator( pattern_char_oracle_matrix )

    wildcard_oracle_matrix = np.multiply( -1, np.identity( int( 2**s ) ) )
    # print(wildcard_oracle_matrix)
    oracles['*'] = quantum_info.Operator( wildcard_oracle_matrix )
    return oracles
