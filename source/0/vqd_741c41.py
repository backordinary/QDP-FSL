# https://github.com/marcobn/Quantum/blob/0b12d7ce39d589e0810c18f7b0eb183120c6ce38/BandTheory/qctb/vqd.py
""" Apply VQD to solve the band-structure of a periodic system. """

import json

import numpy
import qiskit

import qctb.circuit
import qctb.ibmq
import qctb.operator
import qctb.serializeoptimizer
import tightbinding.hamiltonian

# SETUP RNG FOR SEEDING OPTIMIZATIONS
rng = numpy.random.default_rng()
# NOTE: This does NOT affect qiskit simulator.

class VQD:
    """ Calculate and record energy measurements.

    Attributes
    ----------
    H: tightbinding.hamiltonian.Hamiltonian
        the physical system being studied
    QC: qctb.ibmq.QuantumComputer
        encapsulation of settings for running on quantum computer
    OPT: qiskit.aqua.componenets.optimizers.Optimizer
        qiskit aqua Optimizer wrapper
    n: list(int) of length H.D
        number of qubits in each site register (or None for reciprocal space)
    nopt: int
        number of optimization runs for each point
    note: str
        arbitrary descriptive note
    points: list
        each entry corresponds to a specific kpt

    """

    def __init__(self,
        H, QC, OPT, n=None, nopt=1, note="",
        filename=None
    ):
        """ Initialize the VQD object.

        Parameters
        ----------
        H: tightbinding.hamiltonian.Hamiltonian
            the physical system being studied
        QC: qctb.ibmq.QuantumComputer
            encapsulation of settings for running on quantum computer
        OPT: qiskit.aqua.componenets.optimizers.Optimizer
            qiskit aqua Optimizer wrapper
        n: ndarray with length H.D
            number of qubits in each site register (None means reciprocal space)
        nopt: int
            number of optimization runs for each point
        note: str
            arbitrary descriptive note
        filename: string
            filename to append rows of data

        """
        self.H      = H
        self.QC     = QC
        self.OPT    = OPT
        self.n      = n
        self.nopt   = nopt
        self.note   = note
        self.points = []

        self.filename = filename

        self.ansatz = qctb.circuit.ansatz(H.M)

        # REGISTER OPTIMIZATION SETTINGS
        self.OPT_JSON = qctb.serializeoptimizer.toJSON(OPT)
        # NOTE: Some optimizers (eg. SPSA) update settings dynamically,
        #   so we need to remember the original for reproducibility.

        # REAL SPACE APPROACH: PREPARE HAMILTONIAN ONCE
        if self.n is not None:
            self.n = numpy.array(self.n)
            self.op = qctb.operator.RealspaceOperator(H,n, self.ansatz)


    def run(self, ka):
        """ Solve energies at a specific kpt and update points list.

        Parameters
        ----------
        ka: list of float or int
            arbitrary vector in Brillouin zone, if n is None
            quantized point p such that ka = H.kpt(p/2**n), if n is not None

        Returns
        -------
        record: dict
            record appended to points list

        """
        ka = numpy.array(ka)

        # ALIAS CIRCUITS
        ansatz = self.ansatz
        cts = [ansatz]
        # if self.n:
        #     cts.extend([self.qft[i] for i in self.qft])

        # CONVERT ANSATZ AND HAMILTONIAN INTO OPERATOR FLOW
        if self.n is not None:                  # REAL SPACE APPROACH
            p = ka
            ka = self.H.kpt(p/2**self.n)
            energyop = qctb.circuit.bind_p(self.op, p)
        else:                                   # RECIPROCAL SPACE APPROACH
            p = None
            energyop = qctb.operator.ReciprocalOperator(self.H, ka, ansatz)

        # GET STATIC DIAGNOSTIC NUMBERS
        E0 = self.H.E0(ka)                          # ANALYTICAL ENERGY
        depth = qctb.operator.maxdepth(energyop)
        gates = qctb.operator.maxgates(energyop)
        cxcnt = qctb.operator.maxcxcnt(energyop)
        words = qctb.operator.sumwords(energyop)
        abels = qctb.operator.sumabels(energyop)


        # FIND MAX ENERGY (FOR ACCURATE β IN DEFLATION)
        maxlevel = self._optimize(
            -1 * energyop,
            template=self.H.template(ka, self.H.M-1),
        )                                                       # FLIP TO MAX
        maxlevel['E'] = -maxlevel['E']                          # UN-FLIP

        # FIND ENERGY OF EACH LEVEL
        costop = energyop           # GROWS BY OVERLAP MEASUREMENT EACH LEVEL
        levels = []                 # INITIALIZE LEVEL RECORD LIST

        for l in range(self.H.M-1): # SKIP THE LAST POINT, ALREADY IN maxlevel

            # FIND LOWEST ENERGY
            level = self._optimize(
                costop,
                template=self.H.template(ka, l),
            )

            # APPEND OVERLAP CIRCUIT
            if l == 0:
                β = 2*(maxlevel['E'] - level['E'])
                                # DEFLATION AMOUNT: twice the minimum to be safe
                                # Use same β for all levels to treat near bands
            V = qctb.circuit.bind(ansatz, level['x'])   # RECONSTRUCTED CIRCUIT
            '''
            TODO: Higgott et al. recommends optimizing parameterized V†
                such that V†V=I, instead of simply taking V†=V.inverse().
                The former technique should be more robust against gate noise.
                Using V directly is much easier, and requires less time
                on the quantum computer, but we should switch if we can
                find how to hog time for the whole algorithm...
            '''
            costop += β * qctb.operator.OverlapOperator(ansatz, V)
            # TODO: Do we need to merge with nullops on site registers?

            levels.append(level)    # APPEND LEVEL
        levels.append(maxlevel)     # THROW IN THE LAST LEVEL

        # APPEND METADATA TO LEVEL RECORDS
        for l, level in enumerate(levels):
            level["l"]  = l
            level["E0"] = E0[l]

        # APPEND POINT AND RETURN
        point = {
            "ka":       ka.tolist(),
            "p":        None if p is None else p.tolist(),
            "levels":   levels,
            "depth":    depth,
            "gates":    gates,
            "cxcnt":    cxcnt,
            "words":    words,
            "abels":    abels,
        }
        self.points.append(point)
        return point

    def _optimize(self, costop, template=None):
        """ Run optimization and return level record.

        Parameters
        ----------
        costop: qiskit.aqua.operators.SummedOp
            parameterized SummedOp of measurement groups
        x0: list(float)
            initial values to start optimization
        template: string
            string containing patterns for each parameter `x?`, like "{x0} {x1}"
            and nested patterns for `E` and `scale`, like "{{E}} {{scale}}"

        Returns
        -------
        record: dict

        """
        # DEFINE COST FUNCTION WITH TEMPLATING
        def fmt(*x):
            if template is None: return None
            paramdict = {f"x{i}": x[i] for i in range(len(x))}
            return template.format(**paramdict)

        # OPTIMIZE MULTIPLE TIMES; SELECT BEST
        E = float('inf')
        nfev = 0
        for i in range(self.nopt):

            # SELECT INITIAL PARAMETERS
            x0i = rng.random(len(costop.parameters)) * 2*numpy.pi

            # HANDLE BUG IN OPTIMIZATION FOR ZERO PARAMETERS
            if len(x0i) == 0:
                E = self.QC.expectation(
                    costop,
                    template=fmt(),
                    filename=self.filename,
                )
                return {
                    "E":    E,
                    "x":    [],
                    "x0":   [],
                    "nfev": 0,
                }

            # RUN OPTIMIZATION
            xi, Ei, nfevi = self.OPT.optimize(
                len(x0i),
                lambda x: self.QC.expectation(
                    qctb.circuit.bind(costop, x),
                    template=fmt(*x),
                    filename=self.filename,
                ),
                initial_point=x0i,
            )

            # OPTIONAL PRINT STATEMENT
            print (f"Optimization run {i}/{self.nopt-1} complete after {nfevi} function evals.")

            # RESET OPTIMIZER
            self.OPT = qctb.serializeoptimizer.fromJSON(self.OPT_JSON)

            # INCREMENT FUNCTION EVALUATIONS
            nfev += nfevi

            # UPDATE OPTIMUM
            if Ei < E:
                E  =  Ei
                x0 = x0i
                x  =  xi

        # RETURN RECORD
        return {
            "E":    E,
            "x":    list(x),
            "x0":   list(x0),
            "nfev": nfev,
        }

    def save(self, filename):
        """ Save this record to a json file. """
        with open(filename, 'w') as file:
            json.dump({
                "note":         self.note,
                "H":            self.H.toJSON(),
                "QC":           self.QC.toJSON(),
                "OPT":          self.OPT_JSON,
                "n":            None if self.n is None else self.n.tolist(),
                "nopt":         self.nopt,
                "points":       self.points,
            }, file, indent="\t")
        # print ("Saved!")


def load(filename):
    """ Load a VQD object from its serialized json. """
    # LOAD RAW JSON
    with open(filename, 'r') as file:
        _ = json.load(file)

    # DEFAULT VALUES
    if 'nopt' not in _:     _['nopt'] = 1

    # INITIALIZE VQD OBJECT
    vqd = VQD(
        tightbinding.fromJSON(_['H']),
        qctb.ibmq.QuantumComputer.fromJSON(_['QC']),
        qctb.serializeoptimizer.fromJSON(_['OPT']),
        n=_['n'],
        nopt=_['nopt'],
        note=_['note'],
    )

    # ATTACH POINTS
    vqd.points = _['points']

    return vqd
