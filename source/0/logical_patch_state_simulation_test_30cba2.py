# https://github.com/latticesurgery-com/lattice-surgery-compiler/blob/3040c36071062cf1c81ee57551c6de90d258c286/tests/simulation/logical_patch_state_simulation_test.py
# Copyright (C) 2020-2021 - George Watkins, Alex Nguyen, Varun Seshadri
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
from fractions import Fraction
from typing import Dict, List, Tuple

import pytest
import qiskit.opflow as qkop

import lsqecc.logical_lattice_ops.logical_lattice_ops as llops
from lsqecc.pauli_rotations import (  # TODO import as pauli_rotations directly
    circuit,
    rotation,
)
from lsqecc.simulation.lazy_tensor_op import LazyTensorOp
from lsqecc.simulation.logical_patch_state_simulation import (
    LazyTensorPatchSimulator,
    PatchToQubitMapper,
    ProjectiveMeasurement,
    circuit_apply_op_to_qubit,
    proportional_choice,
)
from lsqecc.simulation.qiskit_opflow_utils import bell_pair, to_vector
from tests.simulation.numpy_matrix_assertions import (
    assert_eq_numpy_vectorable,
    assert_eq_numpy_vectors,
)


# Type Alias, looking forward to PEP613 ...
class Outcome(ProjectiveMeasurement.BinaryMeasurementOutcome):
    pass


@pytest.mark.parametrize(
    "circuit, op, idx, desired_state",
    [
        (qkop.Zero, qkop.X, 0, qkop.One),
        (qkop.Zero ^ qkop.Zero, qkop.X, 0, qkop.Zero ^ qkop.One),
        (qkop.Zero ^ qkop.Zero, qkop.X, 1, qkop.One ^ qkop.Zero),
        (qkop.Zero ^ qkop.Zero ^ qkop.Zero, qkop.X, 0, qkop.Zero ^ qkop.Zero ^ qkop.One),
        (qkop.Zero ^ qkop.Zero ^ qkop.Zero, qkop.X, 1, qkop.Zero ^ qkop.One ^ qkop.Zero),
        (qkop.Zero ^ qkop.Zero ^ qkop.Zero, qkop.X, 2, qkop.One ^ qkop.Zero ^ qkop.Zero),
    ],
)
def test_circuit_apply_op_to_qubit(
    circuit: qkop.OperatorBase, op, idx, desired_state: qkop.OperatorBase
):
    circuit_with_op_applied = circuit_apply_op_to_qubit(circuit, op, idx)
    assert_eq_numpy_vectors(
        to_vector(circuit_with_op_applied.eval()), to_vector(desired_state.eval())
    )


class TestProjectiveMeasurement:
    @pytest.mark.parametrize(
        "projector, state, probability",
        [
            ((qkop.I + qkop.Z) / 2, qkop.Zero, 1),
            ((qkop.I - qkop.Z) / 2, qkop.Zero, 0),
            ((qkop.I + qkop.Z) / 2, qkop.Plus, 0.5),
            ((qkop.I - qkop.Z) / 2, qkop.Plus, 0.5),
            ((qkop.I + qkop.X) / 2, qkop.Plus, 1),
            ((qkop.I - qkop.X) / 2, qkop.Plus, 0),
            ((qkop.I + qkop.X) / 2, qkop.Zero, 0.5),
            ((qkop.I - qkop.X) / 2, qkop.Zero, 0.5),
        ],
    )
    def test_borns_rule(self, projector: qkop.OperatorBase, state: qkop.OperatorBase, probability):
        assert ProjectiveMeasurement.borns_rule(projector, state) == pytest.approx(probability)

    @pytest.mark.parametrize(
        "projector, initial_state, desired_state_after_measurement, desired_probability",
        [
            ((qkop.I + qkop.Z) / 2, qkop.Zero, qkop.Zero, 1),
            ((qkop.I + qkop.Z) / 2, qkop.Plus, qkop.Zero, 0.5),
            ((qkop.I - qkop.Z) / 2, qkop.Plus, qkop.One, 0.5),
            ((qkop.I + qkop.X) / 2, qkop.Plus, qkop.Plus, 1),
            ((qkop.I + qkop.X) / 2, qkop.Zero, qkop.Plus, 0.5),
            ((qkop.I - qkop.X) / 2, qkop.Zero, qkop.Minus, 0.5),
        ],
    )
    def test_compute_outcome_state(
        self,
        projector: qkop.OperatorBase,
        initial_state: qkop.OperatorBase,
        desired_state_after_measurement: qkop.OperatorBase,
        desired_probability,
    ):
        actual_state, actual_probability = ProjectiveMeasurement.compute_outcome_state(
            projector, initial_state
        )
        assert actual_probability == pytest.approx(desired_probability)
        assert_eq_numpy_vectors(
            to_vector(desired_state_after_measurement.eval()), to_vector(actual_state)
        )

    @pytest.mark.parametrize(
        "pauli_observable, desired_projectors",
        [
            (qkop.X, ((qkop.I + qkop.X) / 2, (qkop.I - qkop.X) / 2)),
            (
                qkop.I ^ qkop.X,
                (
                    ((qkop.I ^ qkop.I) + (qkop.I ^ qkop.X)) / 2,
                    ((qkop.I ^ qkop.I) - (qkop.I ^ qkop.X)) / 2,
                ),
            ),
            (
                qkop.Z ^ qkop.X,
                (
                    ((qkop.I ^ qkop.I) + (qkop.Z ^ qkop.X)) / 2,
                    ((qkop.I ^ qkop.I) - (qkop.Z ^ qkop.X)) / 2,
                ),
            ),
        ],
    )
    def test_get_projectors_from_pauli_observable(
        self,
        pauli_observable: qkop.OperatorBase,
        desired_projectors: Tuple[qkop.OperatorBase, qkop.OperatorBase],
    ):
        p_0, p_1 = ProjectiveMeasurement.get_projectors_from_pauli_observable(pauli_observable)
        assert_eq_numpy_vectors(to_vector(desired_projectors[0]), to_vector(p_0))
        assert_eq_numpy_vectors(to_vector(desired_projectors[1]), to_vector(p_1))

    @pytest.mark.parametrize(
        "pauli_observable, state_before_measurement, desired_outcome_probability_pairs",
        [
            (qkop.Z, qkop.Zero, [(Outcome(qkop.Zero.eval(), 1), 1)]),
            (
                qkop.Z,
                qkop.Plus,
                [(Outcome(qkop.Zero.eval(), 1), 0.5), (Outcome(qkop.One.eval(), -1), 0.5)],
            ),
            (qkop.X, qkop.Plus, [(Outcome(qkop.Plus.eval(), 1), 1)]),
            (
                qkop.X,
                qkop.Zero,
                [(Outcome(qkop.Plus.eval(), 1), 0.5), (Outcome(qkop.Minus.eval(), -1), 0.5)],
            ),
        ],
    )
    def test_pauli_product_measurement_distribution(
        self,
        pauli_observable: qkop.OperatorBase,
        state_before_measurement: qkop.OperatorBase,
        desired_outcome_probability_pairs: List[Tuple[Outcome, float]],
    ):
        actual_outcome_probability_pairs = (
            ProjectiveMeasurement.pauli_product_measurement_distribution(
                pauli_observable, state_before_measurement
            )
        )
        for i, actual_outcome_probability_pair in enumerate(actual_outcome_probability_pairs):
            actual_outcome, actual_outcome_probability = actual_outcome_probability_pair
            desired_outcome, desired_outcome_probability = desired_outcome_probability_pairs[i]
            assert_eq_numpy_vectors(
                to_vector(desired_outcome.resulting_state),
                to_vector(actual_outcome.resulting_state),
            )
            assert desired_outcome.corresponding_eigenvalue == pytest.approx(
                actual_outcome.corresponding_eigenvalue
            )
            assert desired_outcome_probability == pytest.approx(actual_outcome_probability)


@pytest.mark.parametrize(
    "assoc_data_prob",
    [
        [("a", 0.50), ("b", 0.30), ("c", 0.20)],
        [
            ("a", 1.0),
            ("b", 0),
            ("c", 0),
        ],
    ],
)
def test_proportional_choice(assoc_data_prob):
    NUM_RUNS = 10**5
    DISTRIBUTION_TOLERANCE = 10 ** (-1)

    outcomes: Dict[str, int] = dict([(v, 0) for v, prob in assoc_data_prob])
    for i in range(NUM_RUNS):
        outcomes[proportional_choice(assoc_data_prob)] += 1

    total = sum(outcomes.values())
    for v, prob in assoc_data_prob:
        assert outcomes[v] / total == pytest.approx(prob, rel=DISTRIBUTION_TOLERANCE)


I = circuit.PauliOperator.I  # noqa: E741
X = circuit.PauliOperator.X
Y = circuit.PauliOperator.Y
Z = circuit.PauliOperator.Z


class TestPatchToQubitMapper:
    def test__get_all_operating_patches_just_logical(self):
        c = circuit.PauliOpCircuit(2)
        c.add_pauli_block(circuit.PauliRotation.from_list([X, I], Fraction(1, 2)))

        patch_ids = PatchToQubitMapper._get_all_operating_patches(
            llops.LogicalLatticeComputation(c)
        )
        assert len(patch_ids) == 2
        assert patch_ids[0] != patch_ids[1]

    def test_get_all_operating_patches_logical_with_y_ancilla(self):
        c = circuit.PauliOpCircuit(2)
        c.add_pauli_block(circuit.PauliRotation.from_list([X, I], Fraction(1, 4)))

        logical_computation = llops.LogicalLatticeComputation(c)
        patch_ids = PatchToQubitMapper._get_all_operating_patches(logical_computation)
        assert len(patch_ids) == 3
        assert len(set(patch_ids)) == 3  # check that they are all different

    def test___init__(self):
        c = circuit.PauliOpCircuit(2)
        c.add_pauli_block(circuit.PauliRotation.from_list([X, I], Fraction(1, 2)))
        logical_computation = llops.LogicalLatticeComputation(c)
        mapper = PatchToQubitMapper(logical_computation)
        assert len(mapper.patch_location_to_logical_idx.keys()) == 2
        assert set(mapper.patch_location_to_logical_idx.keys()) == set(
            PatchToQubitMapper._get_all_operating_patches(logical_computation)
        )
        assert set(mapper.patch_location_to_logical_idx.values()) == set(range(2))

    def test_max_num_patches_all_logical(self):
        c = circuit.PauliOpCircuit(2)
        c.add_pauli_block(circuit.PauliRotation.from_list([X, I], Fraction(1, 2)))
        assert PatchToQubitMapper(llops.LogicalLatticeComputation(c)).max_num_patches() == 2

    def test_max_num_patches_with_y_ancilla(self):
        c = circuit.PauliOpCircuit(2)
        c.add_pauli_block(circuit.PauliRotation.from_list([X, I], Fraction(1, 4)))
        assert PatchToQubitMapper(llops.LogicalLatticeComputation(c)).max_num_patches() == 3

    def test_max_num_patches_with_magic_ancilla(self):
        c = circuit.PauliOpCircuit(2)
        c.add_pauli_block(circuit.PauliRotation.from_list([X, I], Fraction(1, 8)))
        assert PatchToQubitMapper(llops.LogicalLatticeComputation(c)).max_num_patches() == 4

    def test_get_uuid(self):
        c = circuit.PauliOpCircuit(2)
        c.add_pauli_block(circuit.PauliRotation.from_list([X, I], Fraction(1, 8)))
        mapper = PatchToQubitMapper(llops.LogicalLatticeComputation(c))
        try:
            mapper.get_uuid(0)
        except Exception as e:
            pytest.fail("get_uuid raised: " + repr(e))
        try:
            mapper.get_uuid(1)
        except Exception as e:
            pytest.fail("get_uuid raised: " + repr(e))
        try:
            mapper.get_uuid(2)
        except Exception as e:
            pytest.fail("get_uuid raised: " + repr(e))
        try:
            mapper.get_uuid(3)
        except Exception as e:
            pytest.fail("get_uuid raised: " + repr(e))

    def test_enumerate_patches_by_index(self):
        c = circuit.PauliOpCircuit(2)
        c.add_pauli_block(circuit.PauliRotation.from_list([X, I], Fraction(1, 8)))
        mapper = PatchToQubitMapper(llops.LogicalLatticeComputation(c))
        iterator = iter(mapper.enumerate_patches_by_index())
        patch_idx, uuid = next(iterator)
        assert patch_idx == 0
        patch_idx, uuid = next(iterator)
        assert patch_idx == 1
        patch_idx, uuid = next(iterator)
        assert patch_idx == 2
        patch_idx, uuid = next(iterator)
        assert patch_idx == 3


class TestLazyTensorPatchSimulator:
    @pytest.mark.parametrize(
        "circuit_op, expected_start_state",
        [
            (rotation.Measurement.from_list([Z]), [qkop.Zero]),
            (rotation.Measurement.from_list([Z, I]), [qkop.Zero, qkop.Zero]),
        ],
    )
    def test_initial_state(
        self,
        circuit_op: rotation.PauliProductOperation,
        expected_start_state: List[qkop.OperatorBase],
    ):
        c = circuit.PauliOpCircuit.from_list([circuit_op])
        sim = LazyTensorPatchSimulator(llops.LogicalLatticeComputation(c))
        assert sim.logical_state.matching_approx_eq_vector(LazyTensorOp(expected_start_state))

    @pytest.mark.parametrize(
        "circuit_op, expected_final_state_possibilities",
        [
            (rotation.Measurement.from_list([Z]), [[qkop.Zero]]),
            (rotation.Measurement.from_list([Z, I]), [[qkop.Zero, qkop.Zero]]),
            (rotation.Measurement.from_list([I, Z]), [[qkop.Zero, qkop.Zero]]),
            (rotation.Measurement.from_list([X]), [[qkop.Plus], [qkop.Minus]]),
            (
                rotation.Measurement.from_list([X, I]),
                [[qkop.Plus, qkop.Zero], [qkop.Minus, qkop.Zero]],
            ),
        ],
    )
    def test_apply_logical_operation(
        self,
        circuit_op: rotation.PauliProductOperation,
        expected_final_state_possibilities: List[List[qkop.OperatorBase]],
    ):
        c = circuit.PauliOpCircuit.from_list([circuit_op])
        computation = llops.LogicalLatticeComputation(c)

        sim = LazyTensorPatchSimulator(computation)
        sim.apply_logical_operation(computation.ops[0])
        assert any(
            [
                sim.logical_state.matching_approx_eq_vector(LazyTensorOp(expected_final_state))
                for expected_final_state in expected_final_state_possibilities
            ]
        )

    @pytest.mark.parametrize(
        "forced_starting_state, circuit_op, expected_final_state_possibilities",
        [
            # disentanglement of the last qubit
            (
                [bell_pair],
                rotation.Measurement.from_list([I, X]),
                [[qkop.Plus, qkop.Plus], [qkop.Minus, qkop.Minus]],
            ),
            # No disentanglement
            (
                [bell_pair],
                rotation.Measurement.from_list([X, I]),
                [[qkop.Plus ^ qkop.Plus], [qkop.Minus ^ qkop.Minus]],
            ),
            # Match between index within operand and
            (
                [qkop.Zero ^ qkop.Zero],
                rotation.PauliRotation.from_list([I, X], Fraction(1, 2)),
                [[qkop.One ^ qkop.Zero]],  # qubit index is in reverse of the ^ operator
            ),
        ],
    )
    def test_circuit_forcing_start_state(
        self,
        forced_starting_state: List[qkop.OperatorBase],
        circuit_op: rotation.PauliProductOperation,
        expected_final_state_possibilities: List[List[qkop.OperatorBase]],
    ):
        c = circuit.PauliOpCircuit.from_list([circuit_op])
        computation = llops.LogicalLatticeComputation(c)

        sim = LazyTensorPatchSimulator(computation)
        sim.logical_state = LazyTensorOp(forced_starting_state)
        sim.apply_logical_operation(computation.ops[0])
        assert any(
            [
                sim.logical_state.matching_approx_eq_vector(LazyTensorOp(expected_final_state))
                for expected_final_state in expected_final_state_possibilities
            ]
        )

    def _create_dummy_simulation_with_starting_state(self, forced_starting_state):
        lazy_forced_starting_state = LazyTensorOp(forced_starting_state)
        # Use a dummy operation to create a LogicalLatticeComputation
        c = circuit.PauliOpCircuit.from_list(
            [
                rotation.PauliRotation.from_list(
                    [X] * lazy_forced_starting_state.get_num_qubits(), Fraction(1, 2)
                )
            ]
        )
        computation = llops.LogicalLatticeComputation(c)
        sim = LazyTensorPatchSimulator(computation)
        sim.logical_state = lazy_forced_starting_state
        return sim

    @pytest.mark.parametrize(
        "forced_starting_state, qubits_to_measure, expected_state",
        [
            ([qkop.Zero], [0], [qkop.Zero]),
            ([qkop.Zero, qkop.One], [0, 1], [qkop.Zero, qkop.One]),
            ([qkop.Zero ^ qkop.One], [0, 1], [qkop.Zero ^ qkop.One]),
            ([qkop.Zero, qkop.One], [1], [qkop.One, qkop.Zero]),
            ([qkop.Zero ^ qkop.One], [0], [qkop.Zero ^ qkop.One]),
            ([qkop.Zero, qkop.One, qkop.Plus], [0], [qkop.Zero, qkop.One, qkop.Plus]),
            ([qkop.Zero, qkop.One, qkop.Plus], [0, 1], [qkop.Zero, qkop.One, qkop.Plus]),
            ([qkop.Zero ^ qkop.One, qkop.Plus], [1], [qkop.Zero ^ qkop.One, qkop.Plus]),
            ([qkop.Zero, qkop.One, qkop.Plus], [1, 2], [qkop.One, qkop.Plus, qkop.Zero]),
            ([qkop.Zero, qkop.One, qkop.Plus], [2], [qkop.Plus, qkop.One, qkop.Zero]),
            ([qkop.Zero ^ qkop.One, qkop.Plus], [2], [qkop.Plus, qkop.Zero ^ qkop.One]),
            ([qkop.Zero, qkop.One, qkop.Plus], [0, 1, 2], [qkop.Zero, qkop.One, qkop.Plus]),
            (
                [bell_pair, qkop.Zero, qkop.One, qkop.Plus],
                [0, 1],
                [bell_pair, qkop.Zero, qkop.One, qkop.Plus],
            ),
            (
                [bell_pair, qkop.Zero, qkop.One, qkop.Plus],
                [1, 2],
                [bell_pair, qkop.Zero, qkop.One, qkop.Plus],
            ),
            (
                [bell_pair, qkop.Zero, qkop.One, qkop.Plus],
                [2, 3],
                [qkop.Zero, qkop.One, bell_pair, qkop.Plus],
            ),
        ],
    )
    def test_bring_active_operands_to_front(
        self,
        forced_starting_state: List[qkop.OperatorBase],
        qubits_to_measure: List[int],
        expected_state: List[qkop.OperatorBase],
    ):
        sim = self._create_dummy_simulation_with_starting_state(forced_starting_state)
        # TODO perform a deeper check on the uuid remapping
        #  (e.g. by checking that the uuid->states match)
        patches_before = sim.mapper.patch_location_to_logical_idx.keys()
        sim.bring_active_operands_to_front(
            llops.MultiBodyMeasurement(
                dict([(sim.mapper.get_uuid(idx), qkop.X) for idx in qubits_to_measure])
            )
        )
        assert expected_state == sim.logical_state.ops
        assert patches_before == sim.mapper.patch_location_to_logical_idx.keys()

    @pytest.mark.parametrize(
        "forced_starting_state, qubits_to_measure, expected_state",
        [
            ([qkop.Zero], [0], [qkop.Zero]),
            ([qkop.Zero, qkop.One], [0, 1], [qkop.Zero ^ qkop.One]),
            ([qkop.Zero ^ qkop.One], [0, 1], [qkop.Zero ^ qkop.One]),
            ([qkop.Zero, qkop.One], [1], [qkop.One, qkop.Zero]),
            ([qkop.Zero ^ qkop.One], [1], [qkop.Zero ^ qkop.One]),
            ([qkop.Zero, qkop.One, qkop.Plus], [0], [qkop.Zero, qkop.One, qkop.Plus]),
            ([qkop.Zero, qkop.One, qkop.Plus], [0, 1], [qkop.Zero ^ qkop.One, qkop.Plus]),
            ([qkop.Zero ^ qkop.One, qkop.Plus], [1], [qkop.Zero ^ qkop.One, qkop.Plus]),
            ([qkop.Zero, qkop.One, qkop.Plus], [1, 2], [qkop.One ^ qkop.Plus, qkop.Zero]),
            ([qkop.Zero, qkop.One, qkop.Plus], [2], [qkop.Plus, qkop.One, qkop.Zero]),
            ([qkop.Zero ^ qkop.One, qkop.Plus], [2], [qkop.Plus, qkop.Zero ^ qkop.One]),
            ([qkop.Zero, qkop.One, qkop.Plus], [0, 1, 2], [qkop.Zero ^ qkop.One ^ qkop.Plus]),
        ],
    )
    def test_align_state_with_pauli_op_to_first_operand(
        self,
        forced_starting_state: List[qkop.OperatorBase],
        qubits_to_measure: List[int],
        expected_state: List[qkop.OperatorBase],
    ):
        sim = self._create_dummy_simulation_with_starting_state(forced_starting_state)
        sim.align_state_with_pauli_op_to_first_operand(
            llops.MultiBodyMeasurement(
                dict([(sim.mapper.get_uuid(idx), qkop.X) for idx in qubits_to_measure])
            )
        )
        assert expected_state == sim.logical_state.ops

    @pytest.mark.parametrize(
        "forced_starting_state, multi_body_measurement_operators, expected_states",
        [
            ([qkop.Zero], [Z], [[qkop.Zero]]),
            ([qkop.Zero, qkop.Zero], [Z, None], [[qkop.Zero, qkop.Zero]]),
            ([qkop.Zero], [X], [[qkop.Plus], [qkop.Minus]]),
            ([qkop.Zero, qkop.Zero], [Z, Z], [[qkop.Zero ^ qkop.Zero]]),
        ],
    )
    def test_apply_logical_operation_multibody_measurement_with_X_gates(
        self,
        forced_starting_state: List[qkop.OperatorBase],
        multi_body_measurement_operators: List[rotation.PauliOperator],
        expected_states: List[List[qkop.OperatorBase]],
    ):
        sim = self._create_dummy_simulation_with_starting_state(forced_starting_state)

        def make_multibody_measurement_for_sim(
            simulator: LazyTensorPatchSimulator,
        ) -> llops.MultiBodyMeasurement:
            return llops.MultiBodyMeasurement(
                dict(
                    [
                        (simulator.mapper.get_uuid(idx), op)
                        for idx, op in enumerate(multi_body_measurement_operators)
                        if op is not None
                    ]
                )
            )

        final_states_alligned_as_input = []
        for expected_state in expected_states:
            sim_clone = self._create_dummy_simulation_with_starting_state(expected_state)
            sim_clone.align_state_with_pauli_op_to_first_operand(
                make_multibody_measurement_for_sim(sim_clone)
            )
            final_states_alligned_as_input.append(sim_clone.logical_state)

        sim.apply_logical_operation(make_multibody_measurement_for_sim(sim))
        print(sim.logical_state)

        print(final_states_alligned_as_input)
        assert any(
            sim.logical_state.matching_approx_eq_vector(expected)
            for expected in final_states_alligned_as_input
        )

    def test_get_separable_states(self):
        c = circuit.PauliOpCircuit.from_list([rotation.Measurement.from_list([X, I])])
        computation = llops.LogicalLatticeComputation(c)

        sim = LazyTensorPatchSimulator(computation)
        sim.logical_state = LazyTensorOp([bell_pair])

        separable_states = sim.get_separable_states()
        assert 0 == len(separable_states)
        sim.apply_logical_operation(computation.ops[0])
        separable_states = sim.get_separable_states()
        assert 2 == len(separable_states)

        print(separable_states)
        try:
            assert_eq_numpy_vectorable(
                qkop.Plus, separable_states[computation.logical_qubit_uuid_map[0]]
            )
        except AssertionError:
            assert_eq_numpy_vectorable(
                qkop.Minus, separable_states[computation.logical_qubit_uuid_map[0]]
            )
        try:
            assert_eq_numpy_vectorable(
                qkop.Plus, separable_states[computation.logical_qubit_uuid_map[1]]
            )
        except AssertionError:
            assert_eq_numpy_vectorable(
                qkop.Minus, separable_states[computation.logical_qubit_uuid_map[1]]
            )
