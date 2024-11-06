# https://github.com/EvanHockings/DMERA/blob/72a39187e8690c9ee372ba8bcbb408f1ba481f39/src/protocol.py
# %%

import math
import random
import mthree  # type: ignore
import numpy as np
import mapomatic as mm  # type: ignore
from copy import deepcopy
from time import time
from typing import Optional, Union, Any
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit, Aer  # type: ignore
from qiskit.circuit.library import XGate  # type: ignore
from qiskit.compiler import transpile  # type: ignore
from qiskit.compiler.scheduler import schedule  # type: ignore
from qiskit.transpiler import PassManager, InstructionDurations  # type: ignore
from qiskit.transpiler.passes import ALAPScheduleAnalysis, PadDynamicalDecoupling, RZXCalibrationBuilder  # type: ignore
from qiskit.transpiler.basepasses import TransformationPass  # type: ignore
from qiskit.dagcircuit import DAGCircuit  # type: ignore
from qiskit.visualization.timeline import draw, IQXStandard  # type: ignore

from protocol_gates import w, w_original, u, u_original, q  # type: ignore

MAIN = __name__ == "__main__"


# %%


def dmera(n: int, d: int) -> tuple[list[list[tuple[int, int]]], list[list[int]]]:
    """
    Args:
        n: Number of scales
        d: Depth at each scale
    Returns:
        circuit: DMERA circuit
        sites_list: List of the sites at each scale
    """
    circuit: list[list[tuple[int, int]]] = []
    sites_list: list[list[int]] = []
    for s in range(n + 1):
        qubits = 3 * 2**s
        sites = [i * (1 << (n - s)) for i in range(qubits)]
        sites_list.append(sites)
        if s != 0:
            for j in range(d):
                if j % 2 == 0:
                    even_sites = [
                        (
                            sites[(2 * i + 0) % (qubits)],
                            sites[(2 * i + 1) % (qubits)],
                        )
                        for i in range(3 * 2 ** (s - 1))
                    ]
                    circuit.append(even_sites)
                else:
                    odd_sites = [
                        (
                            sites[(2 * i + 1) % (qubits)],
                            sites[(2 * i + 2) % (qubits)],
                        )
                        for i in range(3 * 2 ** (s - 1))
                    ]
                    circuit.append(odd_sites)
    return (circuit, sites_list)


# %%


def pcc(
    circuit: list[list[tuple[int, int]]], support: list[int]
) -> list[list[tuple[int, int]]]:
    """
    Args:
        circuit: Circuit
        support: Support of the observable
    Returns:
        pcc_circuit: Past causal cone circuit for the observable
    """
    circuit_reduced: list[list[tuple[int, int]]] = []
    # Store the qubits on which the circuit is supported
    supported_qubits = deepcopy(support)
    # Construct the pcc circuit in reverse order
    for layer in reversed(circuit):
        layer_temp: list[tuple[int, int]] = []
        for gate in layer:
            supported_0 = gate[0] in supported_qubits
            supported_1 = gate[1] in supported_qubits
            # If the gate is in the support of the observable, add it to the circuit, and grow the support appropriately
            if supported_0 or supported_1:
                layer_temp.append(gate)
                if not supported_0:
                    supported_qubits.append(gate[0])
                if not supported_1:
                    supported_qubits.append(gate[1])
        circuit_reduced.append(layer_temp)
    pcc_circuit = list(reversed(circuit_reduced))
    return pcc_circuit


# %%


def qubits_used(
    circuit: list[list[tuple[int, int]]], sites_list: list[list[int]], n: int
) -> tuple[list[int], list[int]]:
    """
    Args:
        circuit: Circuit
        sites_list: List of the sites at each scale
        n: Number of scales
    Returns:
        start_use: The circuit layer at which each qubit starts being used
        stop_use: The circuit layer at which each qubit stops being used
    """
    l = len(circuit)
    circuit_sorted: list[list[int]] = []
    circuit_sorted.append(sites_list[0])
    for layer in circuit:
        layer_sorted = sorted([qubit for gate in layer for qubit in gate])
        circuit_sorted.append(layer_sorted)
    # Mark unused qubits with the default value l
    qubits = 3 * 2**n
    start_use: list[int] = [l] * qubits
    stop_use: list[int] = [l] * qubits
    for qubit in range(qubits):
        # Determine the layers in which the qubit appears
        qubit_used: list[int] = [
            layer_index - 1
            for (layer_index, layer_sorted) in enumerate(circuit_sorted)
            if qubit in layer_sorted
        ]
        # Store the timesteps at which the qubit starts and stops being used.
        if qubit_used:
            start_use[qubit] = qubit_used[0]
            stop_use[qubit] = qubit_used[-1]
    return (start_use, stop_use)


# %%


def track_resets(
    qubits: int,
    l: int,
    reset_length: int,
    reset_configs: int,
    start_use: list[int],
    stop_use: list[int],
    layer_index: int,
    qubit_maps: list[list[int]],
    reset_circuits: list[list[list[int]]],
    resetting_trackers: list[list[int]],
):
    """
    Args:
        qubits: Number of qubits
        l: Number of circuit layers
        reset_length: Time taken to perform the requisite number of resets as a multiple of the time taken for a circuit layer
        reset_configs: Number of reset configurations
        start_use: The circuit layer at which each qubit starts being used
        stop_use: The circuit layer at which each qubit stops being used
        layer_index:
        qubit_maps:
        reset_circuits:
        resetting_trackers:
    Updates:
        qubit_maps
        reset_circuits
        resetting_trackers
    """
    # Determine the qubits that can be reset
    reset_qubits = [
        qubit_index
        for (qubit_index, stop_index) in enumerate(stop_use)
        if stop_index == layer_index
    ]
    for reset_idx in range(reset_configs):
        # Determine the qubits that reset qubits could be used to replace
        reset_allocation_layers = [
            [
                qubit_index
                for (qubit_index, start_index) in enumerate(start_use)
                if start_index == future_layer_index
                and qubit_index not in resetting_trackers[reset_idx]
            ]
            for future_layer_index in range(layer_index + 1 + reset_length, l)
        ]
        # Randomly shuffle the layers
        for layer in reset_allocation_layers:
            random.shuffle(layer)
        # Flatten the layers
        reset_allocation = [
            qubit_index for layer in reset_allocation_layers for qubit_index in layer
        ]
        # Reset qubits and relabel them appropriately with the qubit map
        resets: list[int] = []
        for j in range(min(len(reset_qubits), len(reset_allocation))):
            reset_qubit = reset_qubits[j]
            paired_qubit = reset_allocation[j]
            resets.append(qubit_maps[reset_idx][reset_qubit])
            resetting_trackers[reset_idx].append(paired_qubit)
            qubit_maps[reset_idx][paired_qubit] = qubit_maps[reset_idx][reset_qubit]
            qubit_maps[reset_idx][reset_qubit] = qubits
        reset_circuits[reset_idx].append(resets)
    pass


# %%


def dmera_resets(
    n: int,
    d: int,
    circuit: list[list[tuple[int, int]]],
    sites_list: list[list[int]],
    support: list[int],
    reset_time: Optional[int],
    reset_count: int,
    reset_configs: int,
) -> tuple[
    list[list[tuple[int, int]]],
    list[list[list[int]]],
    list[dict[tuple[int, int], int]],
    list[dict[int, int]],
]:
    """
    Args:
        n: Number of scales
        d: Depth at each scale
        circuit: DMERA circuit
        sites_list: List of the sites at each scale
        support: Support of the observable
        reset_time: Time taken to perform a reset operation as a multiple of the time taken for a circuit layer
        reset_count: Number of reset operations to perform a reset
        reset_configs: Number of random reset configurations to generate
    Returns:
        pcc_circuit: DMERA past causal cone circuit of the observable
        reset_circuits: Qubits which are reset in each layer for each of the reset configurations
        reset_maps: Reset mapping for the qubits upon which the gates in each layer act for each of the reset configurations
        inverse_maps: Inverse mapping for the qubit reset mapping for each of the reset configurations
    """
    # Initialise parameters
    pcc_circuit = pcc(circuit, support)
    l = n * d
    qubits = 3 * 2**n
    assert l == len(pcc_circuit)
    if reset_time is None:
        reset_length = l
    else:
        reset_length = reset_time * reset_count
    (start_use, stop_use) = qubits_used(pcc_circuit, sites_list, n)
    # Initialise the reset trackets for each of the random reset configurations
    qubit_maps: list[list[int]] = [list(range(qubits)) for _ in range(reset_configs)]
    reset_circuits: list[list[list[int]]] = [[[]] for _ in range(reset_configs)]
    resetting_trackers: list[list[int]] = [[] for _ in range(reset_configs)]
    reset_maps: list[dict[tuple[int, int], int]] = [{} for _ in range(reset_configs)]
    # Track resets for initialisation
    track_resets(
        qubits,
        l,
        reset_length,
        reset_configs,
        start_use,
        stop_use,
        -1,
        qubit_maps,
        reset_circuits,
        resetting_trackers,
    )
    for reset_idx in range(reset_configs):
        for qubit in sites_list[0]:
            reset_maps[reset_idx][(-1, qubit)] = qubit_maps[reset_idx][qubit]
    # Determine the resets for the circuit
    for (layer_index, layer) in enumerate(pcc_circuit):
        # Track resets and store the qubit mapping for the gates
        for reset_idx in range(reset_configs):
            for gate in layer:
                for site in gate:
                    if site in resetting_trackers[reset_idx]:
                        resetting_trackers[reset_idx].remove(site)
                    reset_maps[reset_idx][(layer_index, site)] = qubit_maps[reset_idx][
                        site
                    ]
        # Track resets in the layer
        track_resets(
            qubits,
            l,
            reset_length,
            reset_configs,
            start_use,
            stop_use,
            layer_index,
            qubit_maps,
            reset_circuits,
            resetting_trackers,
        )
    # Generate the inverse maps
    inverse_maps: list[dict[int, int]] = [
        {
            qubit: qubit_index
            for (qubit_index, qubit) in enumerate(set(reset_maps[reset_idx].values()))
        }
        for reset_idx in range(reset_configs)
    ]
    return (pcc_circuit, reset_circuits, reset_maps, inverse_maps)


# %%


def theta_evenbly(d: int) -> list[float]:
    """
    Args:
        d: Depth at each scale
    Returns:
        theta_values: Angles supplied in Entanglement renormalization and wavelets by Evenbly and White (2016)
    """
    theta_evenbly_2 = [math.pi / 12, -math.pi / 6]
    theta_evenbly_4 = [
        0.276143653403021,
        0.950326554644286,
        -0.111215262156182,
        -math.pi / 2,
    ]
    theta_evenbly_5 = [
        0.133662134988773,
        -1.311424155804674,
        -0.099557657512352,
        0.717592959416643,
        0.157462489552395,
    ]
    if d == 2:
        theta_values = theta_evenbly_2
    elif d == 4:
        theta_values = theta_evenbly_4
    elif d == 5:
        theta_values = theta_evenbly_5
    else:
        print(
            f"No cached values for theta when d = {d}. Falling back and setting all thetas to be zero."
        )
        theta_values = [0.0] * d
    return theta_values


# %%


def dmera_reset_circuits(
    n: int,
    d: int,
    pcc_circuit: list[list[tuple[int, int]]],
    sites_list: list[list[int]],
    support: list[int],
    theta_values: list[float],
    reset_count: int,
    reset_configs: int,
    reset_circuits: list[list[list[int]]],
    reset_maps: list[dict[tuple[int, int], int]],
    inverse_maps: list[dict[int, int]],
    barriers: bool = False,
    reverse_gate: bool = False,
) -> list[QuantumCircuit]:
    """
    Args:
        n: Number of scales
        d: Depth at each scale
        pcc_circuit: DMERA past causal cone circuit of the observable
        sites_list: List of the sites at each scale
        support: Support of the observable
        theta_values: Gate angles for each layer in each scale
        reset_count: Number of reset operations to perform a reset
        reset_configs: Number of random reset configurations to generate
        reset_circuits: Qubits which are reset in each layer for each of the reset configurations
        reset_maps: Reset mapping for the qubits upon which the gates in each layer act for each of the reset configurations
        inverse_maps: Inverse mapping for the qubit reset mapping for each of the reset configurations
        barriers: Add barriers to the circuit for ease of readability
        reverse_gate: For each gate, reverse which qubit is considered to be the 'first' on which the gate operates
    Returns:
        quantum_circuits: Qiskit circuits implementing the DMERA circuit with reset for each of the reset configurations
    """
    # Determine the qubits from the reset map and set up the circuit
    l = n * d
    classical_register = ClassicalRegister(len(support), "measure")
    quantum_circuits = [
        QuantumCircuit(
            *(
                (
                    QuantumRegister(1, "qubit" + str(qubit))
                    for qubit in list(inverse_maps[reset_idx].keys())
                )
            ),
            classical_register,
        )
        for reset_idx in range(reset_configs)
    ]
    # Initialise the quantum circuit
    q_gate = q().to_gate()
    for reset_idx in range(reset_configs):
        quantum_circuits[reset_idx].append(
            q_gate,
            [
                inverse_maps[reset_idx][reset_maps[reset_idx][(-1, site)]]
                for site in sites_list[0]
            ],
        )
        for qubit in reset_circuits[reset_idx][0]:
            for _ in range(reset_count):
                quantum_circuits[reset_idx].reset(inverse_maps[reset_idx][qubit])
    # Initialise the gates in the quantum circuit
    gate_list = [w_original(theta_values[0]).to_gate()] + [
        u_original(theta_values[layer_index]).to_gate() for layer_index in range(1, d)
    ]
    # Populate the quantum circuit with gates and reset
    for (layer_index, layer) in enumerate(pcc_circuit):
        for reset_idx in range(reset_configs):
            for gate in layer:
                # Append the gate to the circuit
                gate_indices = [
                    inverse_maps[reset_idx][reset_maps[reset_idx][(layer_index, site)]]
                    for site in gate
                ]
                if reverse_gate:
                    gate_indices.reverse()
                quantum_circuits[reset_idx].append(
                    gate_list[layer_index % d], gate_indices
                )
            # Add barriers to the circuit for ease of readability
            if barriers and layer_index != l - 1:
                quantum_circuits[reset_idx].barrier()  # type: ignore
            # Reset the appropriate qubits
            for qubit in reset_circuits[reset_idx][1 + layer_index]:
                for _ in range(reset_count):
                    quantum_circuits[reset_idx].reset(inverse_maps[reset_idx][qubit])
    return quantum_circuits


# %%


def transverse_ising_circuits(
    n: int,
    d: int,
    theta_values: list[float],
    sample_number: int,
    reset_time: Optional[int],
    reset_count: int,
    reset_configs: int,
    print_diagnostics: bool = False,
) -> tuple[
    dict[Union[tuple[int, int], tuple[int, int, int]], list[QuantumCircuit]], list[int]
]:
    """
    Args:
        n: Number of scales
        d: Depth at each scale
        theta_values: Gate angles for each layer in each scale
        sample_number: Number of samples to take
        reset_time: Time taken to perform a reset operation as a multiple of the time taken for a circuit layer
        reset_count: Number of reset operations to perform a reset
        reset_configs: Number of random reset configurations to generate
        print_diagnostics: Print diagnostics
    Returns:
        operator_circuits: Dictionary of lists of Qiskit circuits implementing the DMERA circuit with reset for each of the reset configurations, indexed by the support of the observables
        sites: The local observables of these sites are measured by the circuits
    """
    # Initialise the circuit and variational parameters
    l = n * d
    qubits = 3 * 2**n
    (circuit, sites_list) = dmera(n, d)
    assert l == len(circuit)
    # Generate the supports
    sites = sorted(random.sample(range(qubits), sample_number))
    sites_supports: list[list[Union[tuple[int, int], tuple[int, int, int]]]] = [
        [
            ((site - 1) % qubits, (site + 0) % qubits, (site + 1) % qubits),
            ((site - 1) % qubits, (site + 0) % qubits),
            ((site + 0) % qubits, (site + 1) % qubits),
        ]
        for site in sites
    ]
    flattened_supports: list[Union[tuple[int, int], tuple[int, int, int]]] = sorted(
        list(set([support for supports in sites_supports for support in supports]))
    )
    # Generate the circuits
    qubit_numbers: list[int] = []
    operator_circuits: dict[
        Union[tuple[int, int], tuple[int, int, int]], list[QuantumCircuit]
    ] = {}
    for support in flattened_supports:
        list_support = list(support)
        # Generate the reset circuit parameters
        (pcc_circuit, reset_circuits, reset_maps, inverse_maps) = dmera_resets(
            n,
            d,
            circuit,
            sites_list,
            list_support,
            reset_time,
            reset_count,
            reset_configs,
        )
        # Generate the Qiskit circuit
        quantum_circuits = dmera_reset_circuits(
            n,
            d,
            pcc_circuit,
            sites_list,
            list_support,
            theta_values,
            reset_count,
            reset_configs,
            reset_circuits,
            reset_maps,
            inverse_maps,
        )
        # Add the appropriate measurements to the circuits
        for reset_idx in range(reset_configs):
            # Store the requisite number of qubits
            num_qubits = len(list(set(reset_maps[reset_idx].values())))
            assert num_qubits == quantum_circuits[reset_idx].num_qubits
            qubit_numbers.append(num_qubits)
            # Measure in the appropriate bases
            support_mapped = [
                inverse_maps[reset_idx][reset_maps[reset_idx][(l - 1, qubit)]]
                for qubit in support
            ]
            if len(support) == 2:
                measure_x_qubits = [support_mapped[0], support_mapped[1]]
            elif len(support) == 3:
                measure_x_qubits = [support_mapped[0], support_mapped[2]]
            else:
                raise ValueError("Invalid support")
            quantum_circuits[reset_idx].h(measure_x_qubits)
            quantum_circuits[reset_idx].measure(support_mapped, range(len(support)))
        # Add the circuits to the dictionary
        operator_circuits[support] = quantum_circuits
    # Print a diagnostic
    if print_diagnostics:
        print(
            f"The maximum number of qubits required for any of the DMERA past causal cone circuits is {max(qubit_numbers)}, where the number of scales is {n}, the depth at each scale is {d}, the reset time is {reset_time}, and the reset count is {reset_count}."
        )
    return (operator_circuits, sites)


# %%


def simulator_operators(
    shots: int,
    decomposed_circuits: list[list[QuantumCircuit]],
    transpile_kwargs: dict[str, Any],
    print_diagnostics: bool = False,
) -> list[float]:
    """
    Args:
        shots: Number of shots to measure for each circuit
        decomposed_circuits: List of lists of Qiskit circuits implementing the DMERA circuit with reset for each of the reset configurations with bound parameters for each of the observables
        transpile_kwargs: Keyword arguments for circuit transpiling
        print_diagnostics: Print diagnostics
    Returns:
        operators: Estimated observable operator values
    """
    # If the circuits are being simulated, we don't need to worry about their quality
    backend = transpile_kwargs["backend"]
    start_time = time()
    best_circuits: list[QuantumCircuit] = transpile([circuit_list[0] for circuit_list in decomposed_circuits], **transpile_kwargs)  # type: ignore
    end_time = time()
    if print_diagnostics:
        print(f"Transpiling the circuits took {end_time - start_time} s.")
    # Simulate the circuits
    start_time = time()
    job = backend.run(best_circuits, shots=shots)
    job_results = job.result().get_counts()
    end_time = time()
    if print_diagnostics:
        print(f"Simulating the circuits took {end_time - start_time} s.")
    # Calculate the value of the operators
    counts: list[dict[str, int]] = [
        dict(circuit_counts) for circuit_counts in job_results
    ]
    operators = [
        sum(
            (1 - 2 * (sum(int(bit) for bit in key) % 2)) * count[key]
            for key in count.keys()
        )
        / shots
        for count in counts
    ]
    return operators


# %%


class RemoveDelays(TransformationPass):
    """
    Remove delays from a circuit
    """

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        dag.remove_all_ops_named("delay")  # type: ignore
        return dag


def uhrig_pulse_spacing(k: int) -> list[float]:
    """
    Args:
        k: Pulse sequence length
    Returns:
        spacing: Pulse spacing
    """
    spacing: list[float] = []
    for i in range(k):
        spacing.append(math.sin(math.pi * (i + 1) / (2 * k + 2)) ** 2 - sum(spacing))
    spacing.append(1 - sum(spacing))
    return spacing


def device_operators(
    shots: int,
    decomposed_circuits: list[list[QuantumCircuit]],
    reset_configs: int,
    transpile_configs: int,
    transpile_kwargs: dict[str, Any],
    print_diagnostics: bool = False,
    dynamically_decouple: bool = False,
) -> list[float]:
    """
    Args:
        shots: Number of shots to measure for each circuit
        decomposed_circuits: List of lists of Qiskit circuits implementing the DMERA circuit with reset for each of the reset configurations with bound parameters
        reset_configs: Number of random reset configurations to generate
        transpile_configs: Number of random transpilation configurations to generate
        transpile_kwargs: Keyword arguments for circuit transpiling
        print_diagnostics: Print diagnostics
        dynamically_decouple: Dynamically decouple the circuit
    Returns:
        operators: Estimated observable operator values
    """
    # If the circuits are being run on a device, we need to do a lot to ensure that the results are good
    backend = transpile_kwargs["backend"]
    trimmed_kwargs = deepcopy(transpile_kwargs)
    trimmed_kwargs.pop("layout_method", None)
    trimmed_kwargs.pop("optimization_level", None)
    remove_delays = PassManager(RemoveDelays())
    # rzx_calibrate = PassManager(
    #     RZXCalibrationBuilder(
    #         backend.defaults().instruction_schedule_map,
    #         backend.configuration().qubit_channel_mapping,
    #     )
    # )
    # Set up the dynamical decoupling pass
    if dynamically_decouple:
        k = 8
        dd_sequence = [XGate()] * k
        dd_spacing = uhrig_pulse_spacing(k)
        instruction_durations = InstructionDurations.from_backend(backend)
        alap_schedule = PassManager(ALAPScheduleAnalysis(instruction_durations))  # type: ignore
        dynamically_decouple = PadDynamicalDecoupling(instruction_durations, dd_sequence, qubits=None, spacing=dd_spacing, skip_reset_qubits=True, pulse_alignment=backend.configuration().timing_constraints)  # type: ignore
    assert all(
        [len(circuit_list) == reset_configs for circuit_list in decomposed_circuits]
    )
    start_time = time()
    best_circuits: list[QuantumCircuit] = []
    for circuit_list in decomposed_circuits:
        # Calibrate the circuits for R_ZX pulses
        # calibrated_list: list[QuantumCircuit] = rzx_calibrate.run(circuit_list)  # type: ignore
        # Generate a large set of circuits
        # trial_circuits: list[QuantumCircuit] = transpile(
        #     calibrated_list * transpile_configs, **transpile_kwargs
        # )  # type: ignore
        trial_circuits: list[QuantumCircuit] = transpile(
            circuit_list * transpile_configs, **transpile_kwargs
        )  # type: ignore
        durations = [circuit.duration for circuit in trial_circuits]
        cx_counts = [len(circuit.get_instructions("cx")) for circuit in trial_circuits]
        if any([duration == None for duration in durations]):
            # Choose the circuit with the lowest CX gate count if the circuits do not have durations
            best_arg = np.argmin(cx_counts)  # type: ignore
        else:
            # Choose the circuit with the lowest CX gate count among the reset_configs circuits with the lowest durations
            duration_order = np.argsort(durations)  # type: ignore
            best_arg = duration_order[np.argmin(np.array(cx_counts)[duration_order][0:reset_configs])]  # type: ignore
        # Remove delays and use mapomatic to determine the optimal layout
        deflated_circuit = mm.deflate_circuit(remove_delays.run(trial_circuits[best_arg]))  # type: ignore
        best_layout: tuple[list[int], str, float] = mm.best_overall_layout(deflated_circuit, backend)  # type: ignore
        # Re-transpile the circuit with the optimal layout and then dynamically decouple if appropriate
        if dynamically_decouple:
            best_circuit: QuantumCircuit = dynamically_decouple.run(schedule(transpile(deflated_circuit, initial_layout=best_layout[0], **trimmed_kwargs), backend=backend))  # type: ignore
            if print_diagnostics:
                # TODO: Fix this up when I've sorted out the RZXCalibrationBuilder
                # display(calibrated_list[0].draw("mpl"))  # type: ignore
                # display(trial_circuits[best_arg].draw("mpl"))  # type: ignore
                # display(deflated_circuit.draw("mpl"))  # type: ignore
                # display(draw(best_circuit, style=IQXStandard(**{"formatter.general.fig_width": 40})))  # type: ignore
                pass
        else:
            best_circuit: QuantumCircuit = transpile(
                deflated_circuit,
                initial_layout=best_layout[0],
                optimization_level=3,
                **trimmed_kwargs,
            )  # type: ignore
        best_circuits.append(best_circuit)
    end_time = time()
    if print_diagnostics:
        print(f"Transpiling the circuits took {end_time - start_time} s.")
    # Run the circuits
    start_time = time()
    job = backend.run(best_circuits, shots=shots)
    job_results = job.result().get_counts()
    end_time = time()
    if print_diagnostics:
        print(f"Running the circuits took {end_time - start_time} s.")
    # Calculate the value of the operators using measurement error mitigation
    mitigator = mthree.M3Mitigation(backend)
    measured_qubits: list[list[int]] = [[measurement.qubits[0].index for measurement in best_circuit.get_instructions("measure")] for best_circuit in best_circuits]  # type: ignore
    operators: list[float] = []
    for circuit_counts, qubits in zip(job_results, measured_qubits):
        mitigator.cals_from_system(qubits)  # type: ignore
        mitigated_counts = mitigator.apply_correction(circuit_counts, qubits)  # type: ignore
        operators.append(mitigated_counts.expval())  # type: ignore
    return operators


# %%


def estimate_site_energy(
    n: int,
    shots: int,
    operator_circuits: dict[
        Union[tuple[int, int], tuple[int, int, int]], list[QuantumCircuit]
    ],
    sites: list[int],
    reset_configs: int,
    transpile_configs: int,
    transpile_kwargs: dict[str, Any],
    print_diagnostics: bool = False,
) -> tuple[list[float], list[float]]:
    """
    Args:
        n: Number of scales
        shots: Number of shots to measure for each circuit
        operator_circuits: Dictionary of lists of Qiskit circuits implementing the DMERA circuit with reset for each of the reset configurations, indexed by the support of the observables
        sites: The local observables of these sites are measured by the circuits
        reset_configs: Number of random reset configurations to generate
        transpile_configs: Number of random transpilation configurations to generate
        transpile_kwargs: Keyword arguments for circuit transpiling
        print_diagnostics: Print diagnostics
    Returns:
        sites_energy: Mean energy for each site
        sites_energy_sem: Energy standard error of the mean for each site
    """
    # Re-determine the supports for each site
    qubits = 3 * 2**n
    sites_supports: list[list[Union[tuple[int, int], tuple[int, int, int]]]] = [
        [
            ((site - 1) % qubits, (site + 0) % qubits, (site + 1) % qubits),
            ((site - 1) % qubits, (site + 0) % qubits),
            ((site + 0) % qubits, (site + 1) % qubits),
        ]
        for site in sites
    ]
    flattened_supports: list[Union[tuple[int, int], tuple[int, int, int]]] = sorted(
        list(set([support for supports in sites_supports for support in supports]))
    )
    # Decompose the circuits
    decomposed_circuits: list[list[QuantumCircuit]] = [
        [circuit.decompose() for circuit in operator_circuits[site_support]]
        for site_support in flattened_supports
    ]
    # Transpile and run the circuits
    if transpile_kwargs["backend"].name()[0:3] == "aer":
        operators = simulator_operators(
            shots, decomposed_circuits, transpile_kwargs, print_diagnostics
        )
    else:
        operators = device_operators(
            shots,
            decomposed_circuits,
            reset_configs,
            transpile_configs,
            transpile_kwargs,
            print_diagnostics,
        )
    # Determine the energy at each of the sites
    sites_energy: list[float] = []
    sites_energy_sem: list[float] = []
    for site_index in range(len(sites)):
        site_operators = [
            operators[flattened_supports.index(site_support)]
            for site_support in sites_supports[site_index]
        ]
        energy = site_operators[0] - (site_operators[1] + site_operators[2]) / 2
        energy_sem = math.sqrt(
            (
                (1 - site_operators[0] ** 2)
                + ((1 - site_operators[1] ** 2) + (1 - site_operators[2] ** 2)) / 4
            )
            / shots
        )
        sites_energy.append(energy)
        sites_energy_sem.append(energy_sem)
    return (sites_energy, sites_energy_sem)


# %%


def estimate_energy(
    n: int,
    d: int,
    theta_values: list[float],
    sample_number: int,
    shots: int,
    reset_time: Optional[int],
    reset_count: Optional[int],
    reset_configs: Optional[int],
    transpile_configs: Optional[int],
    transpile_kwargs: dict[str, Any],
    print_diagnostics: bool = False,
) -> tuple[float, float]:
    """
    Args:
        n: Number of scales
        d: Depth at each scale
        theta_values: Gate angles for each layer in each scale
        sample_number: Number of sites to sample the energy from
        shots: Number of shots to measure for each circuit
        reset_time: Time taken to perform a reset operation as a multiple of the time taken for a circuit layer
        reset_count: Number of reset operations to perform a reset
        reset_configs: Number of random reset configurations to generate
        transpile_kwargs: Keyword arguments for circuit transpiling
    Returns:
        energy_mean: Energy per site mean
        energy_sem: Energy per site standard error of the mean
    """
    # Set the optional arguments
    if reset_count is None:
        reset_count = 1
    if reset_configs is None:
        reset_configs = 1
    if transpile_configs is None:
        transpile_configs = 1
    # Generate the circuits
    start_time = time()
    (operator_circuits, sites) = transverse_ising_circuits(
        n, d, theta_values, sample_number, reset_time, reset_count, reset_configs
    )
    end_time = time()
    if print_diagnostics:
        print(f"Generating the circuits took {end_time - start_time} s.")
    # Estimate the energy of the sites
    start_time = time()
    (sites_energy, sites_energy_sem) = estimate_site_energy(
        n,
        shots,
        operator_circuits,
        sites,
        reset_configs,
        transpile_configs,
        transpile_kwargs,
        print_diagnostics,
    )
    end_time = time()
    if print_diagnostics:
        print(f"Estimating the energy took {end_time - start_time} s.")
    # Calculate the mean energy and SEM
    energy_mean: float = np.mean(sites_energy).item()
    energy_sem_shots: float = np.sqrt(
        sum(np.array(sites_energy_sem) ** 2) / (len(sites_energy) ** 2)
    ).item()
    energy_sem_var: float = np.sqrt(np.var(sites_energy) / len(sites_energy)).item()
    energy_sem: float = math.sqrt(energy_sem_shots**2 + energy_sem_var**2)
    if print_diagnostics:
        print(
            f"The average energy per site is {energy_mean:.5f} with a standard error of the mean {energy_sem:.5f}."
        )
    return (energy_mean, energy_sem)


# %%
