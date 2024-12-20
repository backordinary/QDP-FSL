# https://github.com/BQSKit/qfast-qiskit/blob/dd562f809f86d8a2b2633001f78a3ddb0bf3a3af/tests/instantiation/native/qiskit/test_synthesize.py
import numpy    as np
import unittest as ut

from qiskit import *

from qfast import perm
from qfast.instantiation.native.qiskit import QiskitTool

def get_utry ( circ ):
    """Converts a qiskit circuit into a numpy unitary."""

    backend = qiskit.BasicAer.get_backend( 'unitary_simulator' )
    utry = qiskit.execute( circ, backend ).result().get_unitary()
    num_qubits = int( np.log2( len( utry ) ) )
    qubit_order = tuple( reversed( range( num_qubits ) ) )
    P = perm.calc_permutation_matrix( num_qubits, qubit_order )
    return P @ utry @ P.T


def hilbert_schmidt_distance ( X, Y ):
    """Calculates a Hilbert-Schmidt based distance."""

    if X.shape != Y.shape:
        raise ValueError( "X and Y must have same shape." )

    mat = np.matmul( np.transpose( np.conj( X ) ), Y )
    num = np.abs( np.trace( mat ) )
    dem = mat.shape[0]
    return 1 - ( num / dem )


class TestQiskitSynthesize ( ut.TestCase ):

    TOFFOLI = np.asarray(
            [[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
            [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j]] )

    INVALID = np.asarray(
            [[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
            [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j]] )

    def test_qiskit_synthesize_invalid ( self ):
        qtool = QiskitTool()
        self.assertRaises( TypeError, qtool.synthesize, 1 )
        self.assertRaises( TypeError, qtool.synthesize, np.array( [ 0, 1 ] ) )
        self.assertRaises( TypeError, qtool.synthesize, np.array( [ [ [ 0 ] ] ] ) )
        self.assertRaises( TypeError, qtool.synthesize, self.INVALID )

        invalid_utry_matrix = np.copy( self.TOFFOLI )
        invalid_utry_matrix[2][2] = 137.+0.j

        self.assertRaises( TypeError, qtool.synthesize, invalid_utry_matrix )
        self.assertRaises( ValueError, qtool.synthesize, np.identity( 16 )  )

    def test_qiskit_synthesize_valid ( self ):
        qtool = QiskitTool()
        qasm = qtool.synthesize( self.TOFFOLI )
        utry = get_utry( QuantumCircuit.from_qasm_str( qasm ) )
        self.assertTrue( hilbert_schmidt_distance( self.TOFFOLI, utry ) <= 1e-15 )


if __name__ == '__main__':
    ut.main()
