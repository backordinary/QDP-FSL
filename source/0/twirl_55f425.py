# https://github.com/sami-b95/random_circuits/blob/ef9dce33e7de3585b4002af1e9ad5c48786f8bf0/twirl.py
from functools import reduce
import numpy as np
import operator
from qiskit import Aer, ClassicalRegister, execute, QuantumCircuit, QuantumRegister
import RTNI
import scipy.linalg as la
import sympy
import tensornetwork as tn

import converters
import unitary_designs


def numerical_twirl(tensors, output_edges, input_edges, d_value, n_tensor_factors, U):
    """
    Compute the numerical twirl U^{\otimes t}X U^{\dagger\otimes t} of some matrix X.
    
    Parameters
    ----------
    tensors: list
        The list of tensor network nodes from which the matrix X is constructed. For instance, this
        allows X to be expressed as a (matrix) product state. It must have shape
        (d_value, d_value) * n_tensor_factors.
    output_edges: list
        The list of output edges, i.e the edges connected to the line indices of the matrix.
    input_edges: list
        The list of input edges, i.e the edges connected to the column indices of the matrix.
    d_value: int
        The dimension of each tensor product factor.
    n_tensor_factors: int
        The number of tensor product factors (that is, t in the general definition above).
    U: Node
        The node representing the unitary matrix U to twirl with.
    
    Returns
    -------
    array
        The twirled matrix.
    """
    # Makes copies of tensors.
    tensors_copy, edges_copy = tn.copy(tensors)
    input_edges_copy = [edges_copy[input_edge] for input_edge in input_edges]
    output_edges_copy = [edges_copy[output_edge] for output_edge in output_edges]
    U_copies = [tn.copy([U])[0][U] for _ in range(n_tensor_factors)]
    U_conj_copies = [tn.copy([U], conjugate=True)[0][U] for _ in range(n_tensor_factors)]
    # Connect the U^dagger to the column (or "input") index of the matrix.
    for (input_edge_index, input_edge) in enumerate(input_edges_copy):
        input_edge ^ U_conj_copies[input_edge_index][1]
    # Connect the line (or "output") index of the matrix to the U.
    for (output_edge_index, output_edge) in enumerate(output_edges_copy):
        U_copies[output_edge_index][1] ^ output_edge
    # Contract the resulting tensor network.
    result = tn.contractors.greedy(
        list(tensors_copy.values()) + U_copies + U_conj_copies,
        output_edge_order=(
            [U_copy[0] for U_copy in U_copies] +
            [U_conj_copy[0] for U_conj_copy in U_conj_copies]
        )                
    )
    return result

def haar_statistical_average_twirl(tensors, output_edges, input_edges, d_value, n_tensor_factors, n_samples):
    """
    Estimate the Haar average of the twirl of some matrix X by a unitary using statistical sampling.
    
    Parameters
    ----------
    tensors: list
        The list of tensor network nodes from which the matrix X is constructed. For instance, this
        allows X to be expressed as a (matrix) product state.
    output_edges: list
        The list of output edges, i.e the edges connected to the line indices of the matrix.
    input_edges: list
        The list of input edges, i.e the edges connected to the column indices of the matrix.
    d_value: int
        The dimension of each tensor product factor.
    n_tensor_factors: int
        The number of tensor product factors (that is, t in the general definition above).
    n_samples: int
        The number of samples on which to average.
    
    Returns
    -------
    array
        The statistical estimate of the Haar average of the twirled matrix.
    """
    return sum([
        numerical_twirl(
            tensors,
            output_edges,
            input_edges,
            d_value,
            n_tensor_factors,
            tn.Node(unitary_designs.haar_unitary(d_value))
        ).tensor
        for _ in range(n_samples)
    ]) / n_samples

def numerical_haar_average_twirl(tensor, d_value, n_tensor_factors):
    """
    Compute the numerical Haar average of the twirl of some matrix X by a unitary.
    
    Parameters
    ----------
    tensor: Node
        The tensor network node representing the matrix. It must have shape
        (d_value, d_value) * n_tensor_factors.
    d_value: int
        The dimension of each tensor product factor.
    n_tensor_factors: int
        The number of tensor product factors (that is, t in the general definition above).
    
    Returns
    -------
    array
        The numerical Haar average
    """
    # Make copies of tensors.
    tensors_copy, edges_copy = tn.copy([tensor])
    tensor_copy = tensors_copy[tensor]
    output_edges_copy = [edges_copy[output_edge] for output_edge in tensor.edges[:int(len(tensor.edges) / 2)]]
    input_edges_copy = [edges_copy[input_edge] for input_edge in tensor.edges[int(len(tensor.edges) / 2):]]
    # Generate RTNI tensor network graph.
    graph = []
    graph.extend([
        [["X", 1, "in", input_edge_index + 1], ["U*", input_edge_index + 1, "out", 1]]
        for input_edge_index in range(len(input_edges_copy))
    ])
    graph.extend([
        [["U", output_edge_index + 1, "in", 1], ["X", 1, "out", output_edge_index + 1]]
        for output_edge_index in range(len(output_edges_copy))
    ])
    # Symbolically Haar-integrate this tensor network.
    d_symbol = sympy.symbols("d")
    symbolic_average = RTNI.integrateHaarUnitary([graph, 1], ["U", [d_symbol], [d_symbol], d_symbol])
    # Convert the symbolic RTNI average to a numerical TensorNetwork one.
    numerical_average = converters.rtni_to_tn(
        symbolic_average,
        {
            "X": tensor
        },
        d_symbol=d_symbol,
        d_value=d_value,
        n_tensor_factors=n_tensor_factors
    )
    return numerical_average

def statistical_sample_computational_basis(
    basis_labels_left,
    basis_labels_mid_left,
    basis_labels_mid_right,
    basis_labels_right,
    d_value,
    n_samples,
    random_vectors_sampler
):
    """ Compute U_{i_1, i_1'} ... U_{i_t, t_t'} conj(U_{j_1, j_1'}) ... conj(U_{j_t, j_t'}) for
        fixed i_1, ..., i_t, i_1', ..., i_t', j_1, ..., j_t, j_1', ..., j_t' and unitary U
        sampled at random according to a specified algorithm (in fact, only the columns of U
        appearing in the expression are sampled).
        
    Parameters
    ----------
    basis_labels_left: list
        A list specifying i_1, ..., i_t from the definition above.
    basis_labels_mid_left: list
        A list specifying i_1', ..., i_t' from the definition above.
    basis_labels_mid_right: list
        A list specifying j_1', ..., j_t' from the definition above.
    basis_labels_right: list
        A list specifying j_1, ..., j_t from the definition above.
    d_value: int
        The dimension of the random unitary to generate.
    n_samples: int
        The number of random unitaries to draw.
    random_vectors_sampler: function
        A function taking in argument column indices, samples a random unitary U according to
        some algorithm and returns the columns of this unitary specified by the indices (note
        the function needs not sample the whole unitary, but only the required columns, which
        can save some space).
    
    Returns
    -------
    list
        The list of numbers U_{i_1, i_1'} ... U_{i_t, t_t'} conj(U_{j_1, j_1'}) ... conj(U_{j_t, j_t'})
        sampled according to the method described above.
    """
    column_index_to_vector_index = {
        column_index: vector_index
        for vector_index, column_index in enumerate({*basis_labels_mid_left, *basis_labels_mid_right})
    }
    n_vectors = len(column_index_to_vector_index)
    n_tensor_factors = len(basis_labels_left)
    sum_samples = 0
    samples = []
    for _ in range(n_samples):
        vectors = random_vectors_sampler(list(column_index_to_vector_index.keys()))
        sample = reduce(
            operator.mul,
            [
                vectors[basis_labels_left[i], column_index_to_vector_index[basis_labels_mid_left[i]]] *
                np.conj(vectors[basis_labels_right[i], column_index_to_vector_index[basis_labels_mid_right[i]]])
                for i in range(n_tensor_factors)
            ],
            1
        )
        samples.append(sample)
    return samples

def haar_statistical_sample_computational_basis(
    basis_labels_left,
    basis_labels_mid_left,
    basis_labels_mid_right,
    basis_labels_right,
    d_value,
    n_samples
):
    """ Compute U_{i_1, i_1'} ... U_{i_t, t_t'} conj(U_{j_1, j_1'}) ... conj(U_{j_t, j_t'}) for
        fixed i_1, ..., i_t, i_1', ..., i_t', j_1, ..., j_t, j_1', ..., j_t' and Haar-sampled
        unitary U.
        
    Parameters
    ----------
    basis_labels_left: list
        A list specifying i_1, ..., i_t from the definition above.
    basis_labels_mid_left: list
        A list specifying i_1', ..., i_t' from the definition above.
    basis_labels_mid_right: list
        A list specifying j_1', ..., j_t' from the definition above.
    basis_labels_right: list
        A list specifying j_1, ..., j_t from the definition above.
    d_value: int
        The dimension of the random unitary to generate.
    n_samples: int
        The number of random unitaries to draw.
    
    Returns
    -------
    list
        The list of numbers U_{i_1, i_1'} ... U_{i_t, t_t'} conj(U_{j_1, j_1'}) ... conj(U_{j_t, j_t'})
        sampled according to the method described above.
    """
    n_vectors = len({*basis_labels_mid_left, *basis_labels_mid_right})
    n_tensor_factors = len(basis_labels_left)
    haar_random_vectors_sampler = lambda column_indices: la.orth(
        np.random.randn(d_value, n_vectors) + 1j * np.random.randn(d_value, n_vectors)
    )
    return statistical_sample_computational_basis(
        basis_labels_left,
        basis_labels_mid_left,
        basis_labels_mid_right,
        basis_labels_right,
        d_value,
        n_samples,
        haar_random_vectors_sampler
    )

def circuit_statistical_sample_computational_basis(
    basis_labels_left,
    basis_labels_mid_left,
    basis_labels_mid_right,
    basis_labels_right,
    d_value,
    n_samples,
    circuit_sampler
):
    """ Compute U_{i_1, i_1'} ... U_{i_t, t_t'} conj(U_{j_1, j_1'}) ... conj(U_{j_t, j_t'}) for
        fixed i_1, ..., i_t, i_1', ..., i_t', j_1, ..., j_t, j_1', ..., j_t' and unitary U
        is the unitary implemented by a quantum circuit sampled at random.
        
    Parameters
    ----------
    basis_labels_left: list
        A list specifying i_1, ..., i_t from the definition above.
    basis_labels_mid_left: list
        A list specifying i_1', ..., i_t' from the definition above.
    basis_labels_mid_right: list
        A list specifying j_1', ..., j_t' from the definition above.
    basis_labels_right: list
        A list specifying j_1, ..., j_t from the definition above.
    d_value: int
        The dimension of the random unitary to generate.
    n_samples: int
        The number of random unitaries to draw.
    circuit_sampler: function
        A function sampling a random quantum circuit (from some ensemble) in Qiskit format
        (QuantumCircuit class).
    
    Returns
    -------
    list
        The list of numbers U_{i_1, i_1'} ... U_{i_t, t_t'} conj(U_{j_1, j_1'}) ... conj(U_{j_t, j_t'})
        sampled according to the method described above.
    """
    n_vectors = len({*basis_labels_mid_left, *basis_labels_mid_right})
    n_tensor_factors = len(basis_labels_left)
    def random_vectors_sampler(column_indices):
        vectors = np.empty((d_value, len(column_indices)), dtype=complex)
        random_qc = circuit_sampler()
        for iteration, column_index in enumerate(column_indices):
            qr = random_qc.qregs[0]
            init_qc = QuantumCircuit(qr)
            # Set the initial computational basis state
            for bit_index, bit in enumerate(map(int, f"{column_index:b}"[::-1])):
                if bit:
                    init_qc.x(qr[bit_index])
            # Run the full circuit and save the statevector.
            full_qc = init_qc + random_qc
            vectors[:, iteration] = execute(full_qc, backend=Aer.get_backend("statevector_simulator"), shots=n_samples).result().get_statevector()
        return vectors
    return statistical_sample_computational_basis(
        basis_labels_left,
        basis_labels_mid_left,
        basis_labels_mid_right,
        basis_labels_right,
        d_value,
        n_samples,
        random_vectors_sampler
    )

def symbolic_haar_average_bra_ket(n_tensor_factors):
    """ Computes the symbolic average over Haar-random U of
        <left_bra|U^{\otimes n_tensor_factors}|mid_ket><mid_bra|U^{\dagger\otimes n_tensor_factors}|right_ket>.
        The symbolic calculation is performed by the RTNI library.
    
    Parameters
    ----------
    n_tensor_factors: int
        The number of tensor powers of U and U^{\dagger} in the formula above.
    
    Returns
    -------
    tuple
        The first element of the tuple is the symbolic average returned by the RTNI library. The second
        one is the sympy symbol standing for the dimension of the random unitary.
    """
    # Generate RTNI tensor network graph.
    graph = []
    graph.extend([
        [["U*", factor + 1, "in", 1], ["right_ket", 1, "out", factor + 1]]
        for factor in range(n_tensor_factors)
    ])
    graph.extend([
        [["mid_bra", 1, "in", factor + 1], ["U*", factor + 1, "out", 1]]
        for factor in range(n_tensor_factors)
    ])
    graph.extend([
        [["U", factor + 1, "in", 1], ["mid_ket", 1, "out", factor + 1]]
        for factor in range(n_tensor_factors)
    ])
    graph.extend([
        [["left_bra", 1, "out", factor + 1], ["U", factor + 1, "out", 1]]
        for factor in range(n_tensor_factors)
    ])
    # Symbolically Haar-integrate this tensor network.
    d_symbol = sympy.symbols("d")
    symbolic_average = RTNI.integrateHaarUnitary([graph, 1], ["U", [d_symbol], [d_symbol], d_symbol])
    return symbolic_average, d_symbol

def numerical_haar_average_computational_basis(
    basis_labels_left,
    basis_labels_mid_left,
    basis_labels_mid_right,
    basis_labels_right,
    d_value,
    symbolic_average,
    d_symbol
):
    """ Compute the numerical (as opposed to symbolic) average of
        U_{i_1, i_1'} ... U_{i_t, t_t'} conj(U_{j_1, j_1'}) ... conj(U_{j_t, j_t'})
        over Haar-distributed U using equation given in 1902.08539, theorem 3.1.
        
    Parameters
    ----------
    basis_labels_left: list
        A list specifying i_1, ..., i_t from the definition above.
    basis_labels_mid_left: list
        A list specifying i_1', ..., i_t' from the definition above.
    basis_labels_mid_right: list
        A list specifying j_1', ..., j_t' from the definition above.
    basis_labels_right: list
        A list specifying j_1, ..., j_t from the definition above.
    d_value: int
        The dimension of the Haar-random unitary.
    symbolic_average: list
        The symbolic average returned by symbolic_haar_average_bra_ket called with n_tensor_factors
        set to t.
    d_symbol: Symbol
        The dimension symbol returned by symbolic_haar_average_bra_ket called with n_tensor_factors
        set to t.

    Returns
    -------
    complex
        The Haar average of U_{i_1, i_1'} ... U_{i_t, t_t'} conj(U_{j_1, j_1'}) ... conj(U_{j_t, j_t'})
    """
    # Convert the contracted tensor to a numerical value
    value = 0
    for (contracted_edges, coeff) in symbolic_average:
        num_coeff = float(coeff.subs({ d_symbol: d_value }))
        is_zero = False
        for contracted_edge in contracted_edges:
            contracted_edge_start, contracted_edge_end = contracted_edge
            if contracted_edge_start[0] == "left_bra" and contracted_edge_end[0] == "right_ket":
                if basis_labels_left[contracted_edge_start[3] - 1] != basis_labels_right[contracted_edge_end[3] - 1]:
                    is_zero = True
                    break
            elif contracted_edge_start[0] == "mid_bra" and contracted_edge_end[0] == "mid_ket":
                if basis_labels_mid_right[contracted_edge_start[3] - 1] != basis_labels_mid_left[contracted_edge_end[3] - 1]:
                    is_zero = True
                    break
            else:
                raise Exception("unexpected edge in averaged tensor: {}".format(contracted_edge))
        if not is_zero:
            value += num_coeff
    return value