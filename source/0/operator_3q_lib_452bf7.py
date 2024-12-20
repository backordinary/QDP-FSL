# https://github.com/abehersan/QISA/blob/909c779e39e3623a3ef3f3e520c88931a50c5dc3/Quantum_Scrambling_Verification_Test/operator_3q_lib.py
## MODULE WITH OPERATORS FROM:
## "Quantum Scrambling Library" (QSL)
## http://iontrap.umd.edu/wp-content/uploads/2013/10/FiggattThesis.pdf

import numpy as np
from qiskit.quantum_info.operators import Operator
import qiskit

################################## 1. P_op ##################################
#
################################## QSL 5.17 #################################

# The unitary U∗P is a permutation of the unitary U∗ that permutes the first and third input, 
# such that U∗P = P U∗P

# This can also be modeled by simply swapping the inputs of qubits 4 and 6 into the unitary U∗, 
# and swapping them back afterwards.

P_op = Operator([
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1]
                ])

################################## 2. G_op ##################################
#
################################## QSL 5.18 ################################

# Deterministic Teleportation Protocol

G_op = Operator([
                 [0, 0, 0,-1],
                 [0, 1, 0, 0],
                 [0, 0, 1, 0],
                 [-1,0, 0, 0]
                 ])



################################### 3. Us_op ##################################
#
################################## QSL 5.25 ###################################

# The scrambling unitary Us

Us_op = Operator([ 
                [-1, 0, 0,-1, 0,-1,-1, 0], 
                [ 0, 1,-1, 0,-1, 0, 0, 1], 
                [ 0,-1, 1, 0,-1, 0, 0, 1], 
                [ 1, 0, 0, 1, 0,-1,-1, 0],
                [ 0,-1,-1, 0, 1, 0, 0, 1],
                [ 1, 0, 0,-1, 0, 1,-1, 0],
                [ 1, 0, 0,-1, 0,-1, 1, 0],
                [ 0,-1,-1, 0,-1, 0, 0,-1]
                ])

Us_op = 1/2*Us_op

################################### 4. Ucz_op ##################################
#
################################## QSL 5.27 ####################################

# An additional scrambling unitary, UCZ, for use with the Grover protocol

Ucz_op = Operator([[ 0.35355339+0.j,  0.35355339+0.j,  0.35355339+0.j,
                   -0.35355339+0.j,  0.35355339+0.j, -0.35355339+0.j,
                   -0.35355339+0.j, -0.35355339+0.j],
                  [ 0.35355339+0.j, -0.35355339+0.j,  0.35355339+0.j,
                    0.35355339+0.j,  0.35355339+0.j,  0.35355339+0.j,
                   -0.35355339+0.j,  0.35355339+0.j],
                  [ 0.35355339+0.j,  0.35355339+0.j, -0.35355339+0.j,
                    0.35355339+0.j,  0.35355339+0.j, -0.35355339+0.j,
                    0.35355339+0.j,  0.35355339+0.j],
                  [-0.35355339+0.j,  0.35355339+0.j,  0.35355339+0.j,
                    0.35355339+0.j, -0.35355339+0.j, -0.35355339+0.j,
                   -0.35355339+0.j,  0.35355339+0.j],
                  [ 0.35355339+0.j,  0.35355339+0.j,  0.35355339+0.j,
                   -0.35355339+0.j, -0.35355339+0.j,  0.35355339+0.j,
                    0.35355339+0.j,  0.35355339+0.j],
                  [-0.35355339+0.j,  0.35355339+0.j, -0.35355339+0.j,
                   -0.35355339+0.j,  0.35355339+0.j,  0.35355339+0.j,
                   -0.35355339+0.j,  0.35355339+0.j],
                  [-0.35355339+0.j, -0.35355339+0.j,  0.35355339+0.j,
                   -0.35355339+0.j,  0.35355339+0.j, -0.35355339+0.j,
                    0.35355339+0.j,  0.35355339+0.j],
                  [-0.35355339+0.j,  0.35355339+0.j,  0.35355339+0.j,
                    0.35355339+0.j,  0.35355339+0.j,  0.35355339+0.j,
                    0.35355339+0.j, -0.35355339+0.j]],
                 input_dims=(2, 2, 2), output_dims=(2, 2, 2))


################################### 5. Ucs_op ##################################
#
################################## QSL 5.28 ####################################

# Classical Scrambler
# Some unitaries will scramble quantum information non-maximally. This means
# they can transform 1-body information into 3-body information in some bases, but
# not others. Maximally-scrambling unitaries transform 1-body information into 3-
# body information in all bases. 
# One such unitary is the “classical” scrambling unitary UCS,
# which scrambles information in the X and Y bases, but not information in Z.

Ucs_op =  Operator([
                    [1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0,-1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0,-1, 0, 0],
                    [0, 0, 0, 0, 0, 0,-1, 0],
                    [0, 0, 0, 0, 0, 0, 0,-1]
                    ])

############################################################################### 
#
################################## KOMBI P_op - Us_op - P_op #################

U_P = Operator([[-0.5+0.j,  0. +0.j,  0. +0.j, -0.5+0.j,  0. +0.j, -0.5+0.j,
           -0.5+0.j,  0. +0.j],
          [ 0. +0.j,  0.5+0.j, -0.5+0.j,  0. +0.j, -0.5+0.j,  0. +0.j,
            0. +0.j,  0.5+0.j],
          [ 0. +0.j, -0.5+0.j,  0.5+0.j,  0. +0.j, -0.5+0.j,  0. +0.j,
            0. +0.j,  0.5+0.j],
          [ 0.5+0.j,  0. +0.j,  0. +0.j,  0.5+0.j,  0. +0.j, -0.5+0.j,
           -0.5+0.j,  0. +0.j],
          [ 0. +0.j, -0.5+0.j, -0.5+0.j,  0. +0.j,  0.5+0.j,  0. +0.j,
            0. +0.j,  0.5+0.j],
          [ 0.5+0.j,  0. +0.j,  0. +0.j, -0.5+0.j,  0. +0.j,  0.5+0.j,
           -0.5+0.j,  0. +0.j],
          [ 0.5+0.j,  0. +0.j,  0. +0.j, -0.5+0.j,  0. +0.j, -0.5+0.j,
            0.5+0.j,  0. +0.j],
          [ 0. +0.j, -0.5+0.j, -0.5+0.j,  0. +0.j, -0.5+0.j,  0. +0.j,
            0. +0.j, -0.5+0.j]],
         input_dims=(2, 2, 2), output_dims=(2, 2, 2))

############################################################################### 
#
################################## KOMBI P_op - U_star_P - P_op ###############

U_star_P = Operator([[-0.5+0.j,  0. +0.j,  0. +0.j,  0.5+0.j,  0. +0.j,  0.5+0.j,
            0.5+0.j,  0. +0.j],
          [ 0. +0.j,  0.5+0.j, -0.5+0.j,  0. +0.j, -0.5+0.j,  0. +0.j,
            0. +0.j, -0.5+0.j],
          [ 0. +0.j, -0.5+0.j,  0.5+0.j,  0. +0.j, -0.5+0.j,  0. +0.j,
            0. +0.j, -0.5+0.j],
          [-0.5+0.j,  0. +0.j,  0. +0.j,  0.5+0.j,  0. +0.j, -0.5+0.j,
           -0.5+0.j,  0. +0.j],
          [ 0. +0.j, -0.5+0.j, -0.5+0.j,  0. +0.j,  0.5+0.j,  0. +0.j,
            0. +0.j, -0.5+0.j],
          [-0.5+0.j,  0. +0.j,  0. +0.j, -0.5+0.j,  0. +0.j,  0.5+0.j,
           -0.5+0.j,  0. +0.j],
          [-0.5+0.j,  0. +0.j,  0. +0.j, -0.5+0.j,  0. +0.j, -0.5+0.j,
            0.5+0.j,  0. +0.j],
          [ 0. +0.j,  0.5+0.j,  0.5+0.j,  0. +0.j,  0.5+0.j,  0. +0.j,
            0. +0.j, -0.5+0.j]],
         input_dims=(2, 2, 2), output_dims=(2, 2, 2))

############################################################################### 
#
################################## KOMBI P_op - U_cz_P - P_op #################

U_cz_P = Operator([[ 0.35355339+0.j,  0.35355339+0.j,  0.35355339+0.j,
           -0.35355339+0.j,  0.35355339+0.j, -0.35355339+0.j,
           -0.35355339+0.j, -0.35355339+0.j],
          [ 0.35355339+0.j, -0.35355339+0.j,  0.35355339+0.j,
            0.35355339+0.j,  0.35355339+0.j,  0.35355339+0.j,
           -0.35355339+0.j,  0.35355339+0.j],
          [ 0.35355339+0.j,  0.35355339+0.j, -0.35355339+0.j,
            0.35355339+0.j,  0.35355339+0.j, -0.35355339+0.j,
            0.35355339+0.j,  0.35355339+0.j],
          [-0.35355339+0.j,  0.35355339+0.j,  0.35355339+0.j,
            0.35355339+0.j, -0.35355339+0.j, -0.35355339+0.j,
           -0.35355339+0.j,  0.35355339+0.j],
          [ 0.35355339+0.j,  0.35355339+0.j,  0.35355339+0.j,
           -0.35355339+0.j, -0.35355339+0.j,  0.35355339+0.j,
            0.35355339+0.j,  0.35355339+0.j],
          [-0.35355339+0.j,  0.35355339+0.j, -0.35355339+0.j,
           -0.35355339+0.j,  0.35355339+0.j,  0.35355339+0.j,
           -0.35355339+0.j,  0.35355339+0.j],
          [-0.35355339+0.j, -0.35355339+0.j,  0.35355339+0.j,
           -0.35355339+0.j,  0.35355339+0.j, -0.35355339+0.j,
            0.35355339+0.j,  0.35355339+0.j],
          [-0.35355339+0.j,  0.35355339+0.j,  0.35355339+0.j,
            0.35355339+0.j,  0.35355339+0.j,  0.35355339+0.j,
            0.35355339+0.j, -0.35355339+0.j]],
         input_dims=(2, 2, 2), output_dims=(2, 2, 2))

############################################################################### 
#
################################## KOMBI P_op - U_cz_star_P - P_op ############

U_cz_star_P = Operator([[ 0.35355339+0.j,  0.35355339+0.j,  0.35355339+0.j,
           -0.35355339+0.j,  0.35355339+0.j, -0.35355339+0.j,
           -0.35355339+0.j, -0.35355339+0.j],
          [ 0.35355339+0.j, -0.35355339+0.j,  0.35355339+0.j,
            0.35355339+0.j,  0.35355339+0.j,  0.35355339+0.j,
           -0.35355339+0.j,  0.35355339+0.j],
          [ 0.35355339+0.j,  0.35355339+0.j, -0.35355339+0.j,
            0.35355339+0.j,  0.35355339+0.j, -0.35355339+0.j,
            0.35355339+0.j,  0.35355339+0.j],
          [-0.35355339+0.j,  0.35355339+0.j,  0.35355339+0.j,
            0.35355339+0.j, -0.35355339+0.j, -0.35355339+0.j,
           -0.35355339+0.j,  0.35355339+0.j],
          [ 0.35355339+0.j,  0.35355339+0.j,  0.35355339+0.j,
           -0.35355339+0.j, -0.35355339+0.j,  0.35355339+0.j,
            0.35355339+0.j,  0.35355339+0.j],
          [-0.35355339+0.j,  0.35355339+0.j, -0.35355339+0.j,
           -0.35355339+0.j,  0.35355339+0.j,  0.35355339+0.j,
           -0.35355339+0.j,  0.35355339+0.j],
          [-0.35355339+0.j, -0.35355339+0.j,  0.35355339+0.j,
           -0.35355339+0.j,  0.35355339+0.j, -0.35355339+0.j,
            0.35355339+0.j,  0.35355339+0.j],
          [-0.35355339+0.j,  0.35355339+0.j,  0.35355339+0.j,
            0.35355339+0.j,  0.35355339+0.j,  0.35355339+0.j,
            0.35355339+0.j, -0.35355339+0.j]],
         input_dims=(2, 2, 2), output_dims=(2, 2, 2))

###############################################################################

CZ_H_Op = Operator([[ 0.35355339+0.j,  0.35355339+0.j,  0.35355339+0.j,
           -0.35355339+0.j,  0.35355339+0.j, -0.35355339+0.j,
           -0.35355339+0.j, -0.35355339+0.j],
          [ 0.35355339+0.j, -0.35355339+0.j,  0.35355339+0.j,
            0.35355339+0.j,  0.35355339+0.j,  0.35355339+0.j,
           -0.35355339+0.j,  0.35355339+0.j],
          [ 0.35355339+0.j,  0.35355339+0.j, -0.35355339+0.j,
            0.35355339+0.j,  0.35355339+0.j, -0.35355339+0.j,
            0.35355339+0.j,  0.35355339+0.j],
          [-0.35355339+0.j,  0.35355339+0.j,  0.35355339+0.j,
            0.35355339+0.j, -0.35355339+0.j, -0.35355339+0.j,
           -0.35355339+0.j,  0.35355339+0.j],
          [ 0.35355339+0.j,  0.35355339+0.j,  0.35355339+0.j,
           -0.35355339+0.j, -0.35355339+0.j,  0.35355339+0.j,
            0.35355339+0.j,  0.35355339+0.j],
          [-0.35355339+0.j,  0.35355339+0.j, -0.35355339+0.j,
           -0.35355339+0.j,  0.35355339+0.j,  0.35355339+0.j,
           -0.35355339+0.j,  0.35355339+0.j],
          [-0.35355339+0.j, -0.35355339+0.j,  0.35355339+0.j,
           -0.35355339+0.j,  0.35355339+0.j, -0.35355339+0.j,
            0.35355339+0.j,  0.35355339+0.j],
          [-0.35355339+0.j,  0.35355339+0.j,  0.35355339+0.j,
            0.35355339+0.j,  0.35355339+0.j,  0.35355339+0.j,
            0.35355339+0.j, -0.35355339+0.j]],
         input_dims=(2, 2, 2), output_dims=(2, 2, 2))

###############################################################################

CZ_H_Op_t = Operator([[ 0.35355339+0.j,  0.35355339+0.j,  0.35355339+0.j,
            0.35355339+0.j,  0.35355339+0.j,  0.35355339+0.j,
            0.35355339+0.j,  0.35355339+0.j],
          [ 0.35355339+0.j, -0.35355339+0.j,  0.35355339+0.j,
           -0.35355339+0.j, -0.35355339+0.j,  0.35355339+0.j,
           -0.35355339+0.j,  0.35355339+0.j],
          [ 0.35355339+0.j,  0.35355339+0.j,  0.35355339+0.j,
            0.35355339+0.j, -0.35355339+0.j, -0.35355339+0.j,
           -0.35355339+0.j, -0.35355339+0.j],
          [ 0.35355339+0.j, -0.35355339+0.j,  0.35355339+0.j,
           -0.35355339+0.j,  0.35355339+0.j, -0.35355339+0.j,
            0.35355339+0.j, -0.35355339+0.j],
          [ 0.35355339+0.j, -0.35355339+0.j, -0.35355339+0.j,
            0.35355339+0.j, -0.35355339+0.j,  0.35355339+0.j,
            0.35355339+0.j, -0.35355339+0.j],
          [ 0.35355339+0.j,  0.35355339+0.j, -0.35355339+0.j,
           -0.35355339+0.j,  0.35355339+0.j,  0.35355339+0.j,
           -0.35355339+0.j, -0.35355339+0.j],
          [ 0.35355339+0.j, -0.35355339+0.j, -0.35355339+0.j,
            0.35355339+0.j,  0.35355339+0.j, -0.35355339+0.j,
           -0.35355339+0.j,  0.35355339+0.j],
          [ 0.35355339+0.j,  0.35355339+0.j, -0.35355339+0.j,
           -0.35355339+0.j, -0.35355339+0.j, -0.35355339+0.j,
            0.35355339+0.j,  0.35355339+0.j]],
         input_dims=(2, 2, 2), output_dims=(2, 2, 2))