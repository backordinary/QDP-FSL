# https://github.com/cbarange/workshop_cybersecu_quantum/blob/4593adf8d6f8a93b2b2b9ecec53ce770c4ec5142/shor_bench.py
#!/bin/python3

import fractions
import math
import random

from typing import Callable, List, Optional, Sequence, Union

import sympy

import cirq
import time

import numpy as np
from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit.algorithms import Shor






def naive_order_finder(x: int, n: int) -> Optional[int]:

    if x < 2 or n <= x or math.gcd(x, n) > 1:
        raise ValueError(f'Invalid x={x} for modulus n={n}.')
    r, y = 1, x
    while y != 1:
        y = (x * y) % n
        r += 1
    return r


class ModularExp(cirq.ArithmeticOperation):
    
    def __init__(
        self,
        target: Sequence[cirq.Qid],
        exponent: Union[int, Sequence[cirq.Qid]],
        base: int,
        modulus: int,
    ) -> None:
        if len(target) < modulus.bit_length():
            raise ValueError(
                f'Register with {len(target)} qubits is too small for modulus {modulus}'
            )
        self.target = target
        self.exponent = exponent
        self.base = base
        self.modulus = modulus

    def registers(self) -> Sequence[Union[int, Sequence[cirq.Qid]]]:
        return self.target, self.exponent, self.base, self.modulus

    def with_registers(
        self,
        *new_registers: Union[int, Sequence['cirq.Qid']],
    ) -> 'ModularExp':
        if len(new_registers) != 4:
            raise ValueError(
                f'Expected 4 registers (target, exponent, base, '
                f'modulus), but got {len(new_registers)}'
            )
        target, exponent, base, modulus = new_registers
        if not isinstance(target, Sequence):
            raise ValueError(f'Target must be a qubit register, got {type(target)}')
        if not isinstance(base, int):
            raise ValueError(f'Base must be a classical constant, got {type(base)}')
        if not isinstance(modulus, int):
            raise ValueError(f'Modulus must be a classical constant, got {type(modulus)}')
        return ModularExp(target, exponent, base, modulus)

    def apply(self, *register_values: int) -> int:
        assert len(register_values) == 4
        target, exponent, base, modulus = register_values
        if target >= modulus:
            return target
        return (target * base ** exponent) % modulus

    def _circuit_diagram_info_(
        self,
        args: cirq.CircuitDiagramInfoArgs,
    ) -> cirq.CircuitDiagramInfo:
        assert args.known_qubits is not None
        wire_symbols: List[str] = []
        t, e = 0, 0
        for qubit in args.known_qubits:
            if qubit in self.target:
                if t == 0:
                    if isinstance(self.exponent, Sequence):
                        e_str = 'e'
                    else:
                        e_str = str(self.exponent)
                    wire_symbols.append(f'ModularExp(t*{self.base}**{e_str} % {self.modulus})')
                else:
                    wire_symbols.append('t' + str(t))
                t += 1
            if isinstance(self.exponent, Sequence) and qubit in self.exponent:
                wire_symbols.append('e' + str(e))
                e += 1
        return cirq.CircuitDiagramInfo(wire_symbols=tuple(wire_symbols))


def make_order_finding_circuit(x: int, n: int) -> cirq.Circuit:

    L = n.bit_length()
    target = cirq.LineQubit.range(L)
    exponent = cirq.LineQubit.range(L, 3 * L + 3)
    return cirq.Circuit(
        cirq.X(target[L - 1]),
        cirq.H.on_each(*exponent),
        ModularExp(target, exponent, x, n),
        cirq.qft(*exponent, inverse=True),
        cirq.measure(*exponent, key='exponent'),
    )


def read_eigenphase(result: cirq.Result) -> float:

    exponent_as_integer = result.data['exponent'][0]
    exponent_num_bits = result.measurements['exponent'].shape[1]
    return float(exponent_as_integer / 2 ** exponent_num_bits)


def quantum_order_finder(x: int, n: int) -> Optional[int]:

    if x < 2 or n <= x or math.gcd(x, n) > 1:
        raise ValueError(f'Invalid x={x} for modulus n={n}.')

    circuit = make_order_finding_circuit(x, n)
    result = cirq.sample(circuit)
    eigenphase = read_eigenphase(result)
    f = fractions.Fraction.from_float(eigenphase).limit_denominator(n)
    if f.numerator == 0:
        return None  # coverage: ignore
    r = f.denominator
    if x ** r % n != 1:
        return None  # coverage: ignore
    return r


def find_factor_of_prime_power(n: int) -> Optional[int]:
    
    for k in range(2, math.floor(math.log2(n)) + 1):
        c = math.pow(n, 1 / k)
        c1 = math.floor(c)
        if c1 ** k == n:
            return c1
        c2 = math.ceil(c)
        if c2 ** k == n:
            return c2
    return None


def find_factor(
    n: int, order_finder: Callable[[int, int], Optional[int]], max_attempts: int = 30
) -> Optional[int]:
    """Returns a non-trivial factor of composite integer n.

    Args:
        n: integer to factorize,
        order_finder: function for finding the order of elements of the
            multiplicative group of integers modulo n,
        max_attempts: number of random x's to try, also an upper limit
            on the number of order_finder invocations.

    Returns:
        Non-trivial factor of n or None if no such factor was found.
        Factor k of n is trivial if it is 1 or n.
    """
    if sympy.isprime(n):
        return None
    if n % 2 == 0:
        return 2
    c = find_factor_of_prime_power(n)
    if c is not None:
        return c
    for _ in range(max_attempts):
        x = random.randint(2, n - 1)
        c = math.gcd(x, n)
        if 1 < c < n:
            return c  # coverage: ignore
        r = order_finder(x, n)
        if r is None:
            continue  # coverage: ignore
        if r % 2 != 0:
            continue  # coverage: ignore
        y = x ** (r // 2) % n
        assert 1 < y < n
        c = math.gcd(y - 1, n)
        if 1 < c < n:
            return c
    return None  # coverage: ignore


def main( n: int, order_finder: Callable[[int, int], Optional[int]] = naive_order_finder, ):
    
  d = find_factor(n, order_finder)

  if d is None:
    return "" #  print(f'No non-trivial factor of {n} found. It is probably a prime.')
  else:
    return d #print(f'({d},{round(n/d)}) are a non-trivial factors of {n}')

    # assert 1 < d < n
    # assert n % d == 0


def shor_qiskit(number):
  N = int(number)
  backend = Aer.get_backend('aer_simulator')
  quantum_instance = QuantumInstance(backend, shots=1024)
  shor = Shor(quantum_instance=quantum_instance)
  result = shor.factor(N)
  return result
  

if __name__ == '__main__':
  # coverage: ignore
  ORDER_FINDERS = { 'naive': naive_order_finder, 'quantum': quantum_order_finder }
  
  primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
  print("number;factors;compute_time;algo;prime")
  for number in range(3,100):
    if number%2==0:
      continue
    
    # - naive -
    start_time = time.time()
    res = main(n=number, order_finder=ORDER_FINDERS["naive"])
    compute_time = "%s" % (time.time() - start_time)
    print(f"{number};({res},{round(number/res) if type(res) ==int else ''});{compute_time};naive;{'prime' if number in primes else 'non-prime'}")

    # - qiskit -
    start_time = time.time()
    res = shor_qiskit(number)
    compute_time = "%s" % (time.time() - start_time)
    factors = map(str,res.factors)
    print(f"{number};({'' if len(res.factors)==0 else ','.join(factors)});{compute_time};qiskit;{'prime' if number in primes else 'non-prime'}")

    # - cirq -
    start_time = time.time()
    res = main(n=number, order_finder=ORDER_FINDERS["quantum"])
    compute_time = "%s" % (time.time() - start_time)
    print(f"{number};({res},{round(number/res) if type(res) ==int else ''});{compute_time};cirq;{'prime' if number in primes else 'non-prime'}")

  
