# https://github.com/FehintolaObafemi/Shor-s-Algorithm/blob/629837eb5987b8d6ebf9469397529684b6478724/main.py
# shors.py
from qiskit import IBMQ
from qiskit.aqua import QuantumInstance
from shors import Shors
import numpy as np

IBMQ.enable_account(
    '8cd11e82260ef3b66215a7699d2ae154e1c7cd2b94cea401741b2abe7e7f39fc2258b45c4a31a5bd6c47de7f0e10c8f5c3abd0bdd23b0abb81cea1d56d0c6032')
provider = IBMQ.get_provider(hub='ibm-q')

backend = provider.get_backend('ibmq_qasm_simulator')  # Specifies the quantum device

print('\n Shors Algorithm')
print('--------------------')
print('\nExecuting...\n')

N = int(input("Enter a number: "))
a = 3
factors = Shors(N, a)  # Function to run Shor's algorithm where 21 is the integer to be factored
N2 = int(input("Enter another number: "))

xvals = factors.order()
yvals = [np.mod(a ** x, N) for x in xvals]

# getting possible period values
r = yvals[1:].index(1) + 1
print("possible period value: ", r)

check = a**(r/2) + 1
print(check)
if check != 0:
    print("The prime factors of", N, "are", factors.gcd())

if N2 > 15:
    factors2 = Shors(N2, a)
    xvals2 = factors2.order()
    yvals2 = [np.mod(a ** x, N2) for x in xvals2]

    r = yvals2[1:].index(1) + 1
    print("possible period value: ", r)

    check = a ** (r / 2) + 1
    print(check)
    if check != 0:
        print("The prime factors of", N2, "are", factors2.gcd())
