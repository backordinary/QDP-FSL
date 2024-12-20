# https://github.com/latticesurgery-com/lattice-surgery-compiler/blob/3040c36071062cf1c81ee57551c6de90d258c286/src/lsqecc/simulation/qubit_state.py
# Copyright (C) 2020-2021 - George Watkins and Alex Nguyen
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301
# USA

from __future__ import annotations

import cmath
import enum
from typing import TYPE_CHECKING, Optional, Tuple

import qiskit.opflow as qkop

import lsqecc.simulation.qiskit_opflow_utils as qkutil

if TYPE_CHECKING:
    from lsqecc.pauli_rotations import PauliOperator


class ActivityType(enum.Enum):
    Unitary = "Unitary"
    Measurement = "Measurement"


class QubitActivity:
    # TODO refactor to simplify here and remove QubitInMeasurementActivity
    def __init__(self, op: PauliOperator, activity_type: ActivityType):
        self.op = op
        self.activity_type = activity_type


class QubitState:
    def ket_repr(self):
        raise Exception("Method not implemented")

    def compose_operator(self, op: PauliOperator):
        return self  # Do nothing

    def apply_measurement(self, basis: PauliOperator):
        raise Exception("Method not implemented")


class SymbolicState(QubitState):
    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def ket_repr(self):
        return self.name

    def compose_operator(self, op: PauliOperator):
        return ActiveState(
            self, DefaultSymbolicStates.UnknownState, QubitActivity(op, ActivityType.Unitary)
        )

    def apply_measurement(self, basis: PauliOperator):
        return ActiveState(
            self, DefaultSymbolicStates.UnknownState, QubitActivity(basis, ActivityType.Measurement)
        )


class ActiveState(QubitState):
    def __init__(self, prev: QubitState, next: QubitState, activity: QubitActivity):
        self.prev = prev
        self.next = next
        self.activity = activity

    def ket_repr(self):
        if self.activity.activity_type == ActivityType.Measurement:
            return "Measuring {:s}:\n{:s}".format(str(self.activity.op), self.next.ket_repr())
        if self.activity.activity_type == ActivityType.Unitary:
            if isinstance(self.prev, EntangledState):
                return "Apply {:s}\n to entangled\n state".format(self.activity.op)
            elif len(self.prev.ket_repr() + self.prev.ket_repr()) < 10:
                # Compact printing
                return str(self.activity.op) + self.prev.ket_repr() + " = " + self.next.ket_repr()
            else:
                return "{:s}({:s})\n={:s}".format(
                    str(self.activity.op), self.prev.ket_repr(), self.next.ket_repr()
                )


class EntangledState(SymbolicState):
    def __init__(self):
        super().__init__("Entangled")

    def ket_repr(self):
        return "Entangled"


class DefaultSymbolicStates:
    Zero = SymbolicState("|0>")
    One = SymbolicState("|1>")
    Plus = SymbolicState("|+>")
    Minus = SymbolicState("|->")
    YPosEigenState = SymbolicState("|(Y+)>")
    YNegEigenState = SymbolicState("|(Y-)>")
    Magic = SymbolicState("|m>")
    UnknownState = SymbolicState("|?>")

    @staticmethod
    def from_amplitudes(zero_ampl: complex, one_ampl: complex) -> SymbolicState:
        # normalize
        mag = cmath.sqrt(zero_ampl * zero_ampl.conjugate() + one_ampl * one_ampl.conjugate())
        zero_ampl /= mag
        one_ampl /= mag

        # Set global phase to 0
        gphase = cmath.phase(zero_ampl)
        zero_ampl /= cmath.exp(1j * gphase)
        one_ampl /= cmath.exp(1j * gphase)

        def close(a, b):
            return cmath.isclose(a, b, rel_tol=10 ** (-6))

        if close(zero_ampl, 0):
            return DefaultSymbolicStates.One
        if close(one_ampl, 0):
            return DefaultSymbolicStates.Zero
        if close(zero_ampl, cmath.sqrt(2) / 2):
            if close(one_ampl, cmath.sqrt(2) / 2):
                return DefaultSymbolicStates.Plus
            if close(one_ampl, -cmath.sqrt(2) / 2):
                return DefaultSymbolicStates.Minus
            if close(one_ampl, cmath.sqrt(2) / 2 * 1j):
                return DefaultSymbolicStates.YPosEigenState
            if close(one_ampl, -cmath.sqrt(2) / 2 * 1j):
                return DefaultSymbolicStates.YNegEigenState
            if close(one_ampl, cmath.sqrt(2) / 2 * cmath.exp(1j * cmath.pi / 4)):
                return DefaultSymbolicStates.Magic

        return SymbolicState("{:.2f}|0>\n{:+.2f}|1>".format(zero_ampl.real, one_ampl.real))

    @staticmethod
    def get_amplitudes(s: SymbolicState) -> Tuple[complex, complex]:
        """Returns in order the zero amplitude and the one amplitude"""
        if s == DefaultSymbolicStates.Zero:
            return 1, 0
        if s == DefaultSymbolicStates.One:
            return 0, 1
        elif s == DefaultSymbolicStates.Plus:
            return cmath.sqrt(2) / 2, cmath.sqrt(2) / 2
        elif s == DefaultSymbolicStates.Minus:
            return cmath.sqrt(2) / 2, -cmath.sqrt(2) / 2
        elif s == DefaultSymbolicStates.YPosEigenState:
            return cmath.sqrt(2) / 2, cmath.sqrt(2) / 2 * 1j
        elif s == DefaultSymbolicStates.YNegEigenState:
            return cmath.sqrt(2) / 2, -cmath.sqrt(2) / 2 * 1j
        elif s == DefaultSymbolicStates.Magic:
            return cmath.sqrt(2) / 2, cmath.exp(1j * cmath.pi / 4) / cmath.sqrt(2)
        raise NotImplementedError

    @staticmethod
    def from_state_fn(state: qkop.StateFn) -> SymbolicState:
        alpha, beta = qkutil.to_vector(state)
        return DefaultSymbolicStates.from_amplitudes(alpha, beta)

    @staticmethod
    def from_maybe_state_fn(state: Optional[qkop.StateFn]) -> SymbolicState:
        return (
            DefaultSymbolicStates.from_state_fn(state)
            if state is not None
            else DefaultSymbolicStates.UnknownState
        )
