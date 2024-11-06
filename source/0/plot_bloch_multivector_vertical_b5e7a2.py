# https://github.com/michelangelo21/Qwint/blob/612a2ef145478444130863f78d494bb023588884/plot_bloch_multivector_vertical.py
import qiskit
from qiskit.visualization import plot_bloch_vector
from qiskit.utils.deprecation import deprecate_arguments
from qiskit.utils import optionals as _optionals
from qiskit.visualization.utils import (
    _bloch_multivector_data,
    _paulivec_data,
    matplotlib_close_if_inline,
)

# [docs]@deprecate_arguments({"rho": "state"})
@_optionals.HAS_MATPLOTLIB.require_in_call
def plot_bloch_multivector_vertical(
    state, title="", figsize=None, *, rho=None, reverse_bits=False, filename=None
):
    """Plot the Bloch sphere.

    Plot a sphere, axes, the Bloch vector, and its projections onto each axis.

    Args:
        state (Statevector or DensityMatrix or ndarray): an N-qubit quantum state.
        title (str): a string that represents the plot title
        figsize (tuple): Has no effect, here for compatibility only.
        reverse_bits (bool): If True, plots qubits following Qiskit's convention [Default:False].

    Returns:
        matplotlib.Figure:
            A matplotlib figure instance.

    Raises:
        MissingOptionalLibraryError: Requires matplotlib.
        VisualizationError: if input is not a valid N-qubit state.

    Example:
        .. jupyter-execute::

            from qiskit import QuantumCircuit
            from qiskit.quantum_info import Statevector
            from qiskit.visualization import plot_bloch_multivector
            %matplotlib inline

            qc = QuantumCircuit(2)
            qc.h(0)
            qc.x(1)

            state = Statevector.from_instruction(qc)
            plot_bloch_multivector(state)
    """
    from matplotlib import pyplot as plt

    # Data
    bloch_data = (
        _bloch_multivector_data(state)[::-1] if reverse_bits else _bloch_multivector_data(state)
    )
    num = len(bloch_data)
    width, height = plt.figaspect(1 / num)
    fig = plt.figure(figsize=(width, height))
    for i in range(num):
        pos = num - 1 - i if reverse_bits else i
        ax = fig.add_subplot(num, 1, i + 1, projection="3d")
        plot_bloch_vector(bloch_data[i], ax=ax, figsize=figsize)
    # fig.suptitle(title, fontsize=16, y=1.01)
    matplotlib_close_if_inline(fig)
    if filename is None:
        return fig
    else:
        return fig.savefig(filename)
