# https://github.com/rkruegs123/qcircuit-optimization/blob/8e2fcba51dbbf754d84658f6de7974ac02a93671/ga_optimizers.py
import random
from random import shuffle
from tqdm import tqdm
from base_optimizer import Optimizer
import pdb
from functools import partial
from copy import deepcopy

# pytket
from pytket.extensions.pyzx import pyzx_to_tk, tk_to_pyzx
from pytket import OpType
from pytket.passes import RemoveRedundancies, PauliSimp, CliffordSimp, KAKDecomposition, \
    RebasePyZX
from pytket.qasm import circuit_to_qasm_str

# qiskit
from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import BasicSwap, LookaheadSwap, StochasticSwap, Optimize1qGates, \
    Optimize1qGatesDecomposition, CXCancellation

from congruences import is_pivot_edge, pivot_cong, is_lc_vertex, lc_cong, apply_rand_pivot, \
    apply_rand_lc
from utilities import to_graph_like, g_score, c_score

import sys
sys.path.append('../pyzx')
import pyzx as zx
from pyzx import basicrules

class Mutant:
    def __init__(self, c, g):
        self.c_orig = c
        self.c_curr = c
        self.g_curr = g.copy()
        # FIXME: Maybe Mutant shouldn't own its own score
        self.score = None
        self.dead = False # no more actions can be applied to it


def default_score(m):
    return c_score(m.c_curr)
    # return 4 * m.c_curr.twoqubitcount() + m.c_curr.tcount()

class GeneticOptimizer(Optimizer):
    def __init__(self, actions, score=default_score, quiet=True):
        # FIXME: Action should accept circ and graph and return (success, new circ and graph). Allows for actions that act on both graphs and circuits.
        self.actions = actions
        # self.n_gens = n_generations
        # self.n_mutants = n_mutants

        self.score = score # function that maps Circuit -> Double
        self.quiet = quiet

    # FIXME: multi-thread (as long as no actions are multithreaded)
    def mutate(self):
        for m in tqdm(self.mutants, desc="Mutating", disable=self.quiet):
            # Note: actions have to look for their own "matches".
            success = False
            shuffle(self.actions)
            for a in self.actions:
                success, (c_new, g_new) = a(m.c_curr, m.g_curr)
                if success:
                    # FIXME: If pass by reference, don't have to do this. Indeed, may want to pass copy
                    m.c_curr = c_new.to_basic_gates()
                    m.g_curr = g_new.copy() # copy() to make vertices consecutive
                    m.score = self.score(m)
                    break
            if not success:
                m.dead = True

    def update_scores(self):
        for m in self.mutants:
            m.score = self.score(m) # FIXME: Bad idiom

    def select(self, method="tournament"):
        if method == "tournament":
            new_mutants = list()
            for _ in range(self.n_mutants):
                m1, m2 = random.sample(self.mutants, 2)
                if m1.dead:
                    new_mutants.append(m2)
                elif m1.score < m2.score: # Reminder: lower is better
                    new_mutants.append(m1)
                else:
                    new_mutants.append(m2)
            self.mutants = [deepcopy(m) for m in new_mutants]
        elif method == "top_half":
            ms_tmp = self.mutants.copy()
            ms_tmp = sorted(ms_tmp, key=lambda m: m.score)
            n = self.n_mutants // 2
            top_half = ms_tmp[:n]
            if self.n_mutants % 2 == 0:
                self.mutants = deepcopy(top_half) * 2
            else:
                self.mutants = deepcopy(top_half) * 2 + deepcopy([ms_tmp[0]])
        elif method == "top_n":
            n = 10
            ms_tmp = self.mutants.copy()
            ms_tmp = sorted(ms_tmp, key=lambda m: m.score)
            top_n = ms_tmp[:n]
            fact = self.n_mutants // n
            r = self.n_mutants - fact * n
            self.mutants = top_n * fact + top_n[:r]
            # Hack to not mess up memory addresses...
            self.mutants = [deepcopy(m) for m in self.mutants]
        else:
            raise RuntimeError(f"[select] Unknown selection method {method}")

    def _optimize(self, c):
        _, c_opt = self.evolve(c)
        return c_opt

    # def evolve(self, c, n_mutants, n_generations):
    def evolve(self, g, n_mutants, n_generations):
        self.n_mutants = n_mutants
        self.n_gens = n_generations

        self.g_orig = g.copy()
        to_graph_like(self.g_orig)
        self.c_orig = zx.extract_circuit(self.g_orig.copy()).to_basic_gates()
        self.c_orig = zx.basic_optimization(self.c_orig)

        # self.c_orig = c
        # self.g_orig = c.to_graph()
        # to_graph_like(self.g_orig)
        # self.mutants = [Mutant(c, self.g_orig) for _ in range(self.n_mutants)]
        self.mutants = [Mutant(self.c_orig, self.g_orig) for _ in range(self.n_mutants)]

        self.update_scores()
        best_mutant = min(self.mutants, key=lambda m: m.score) # FIXME: Check if this assignment is by reference or value
        best_score = best_mutant.score

        gen_scores = [best_score]
        best_scores = [best_score]
        for i in tqdm(range(self.n_gens), desc="Generations", disable=self.quiet):
            n_unique_mutants = len(list(set([id(m) for m in self.mutants])))
            assert(n_unique_mutants == self.n_mutants)

            self.mutate()
            # self.update_scores()
            best_in_gen = min(self.mutants, key=lambda m: m.score)
            gen_scores.append(best_in_gen.score) # So that if we never improve, we see each generation. Because our actions might all rely on extracting, we may never improve on the original
            if best_in_gen.score < best_score:
                best_mutant = deepcopy(best_in_gen)
                best_score = best_in_gen.score

            best_scores.append(best_score)
            if all([m.dead for m in self.mutants]):
                print("[_optimize] stopping early -- all mutants are dead")
                break

            self.select()

        return best_scores, gen_scores, best_mutant.c_curr

    @property
    def name(self):
        return f"Genetic"

    @property
    def desc(self):
        return "A generic genetic algorithm with tournament selection. FIXME: incorporate an identifier for actions, etc"



##### Library of ACTIONS for GeneticOptimizer
# FIXME: Again, note that all this passing around of c and g may be unnecessary

### Congruence actions

# IMPORTANT: Usage of these relies on a scoring function that does the full_reduce itself. Here
# we don't always full_reduce because we don't want to limit the search space.
def rand_pivot(c, g, reduce_prob=0.1):
    g_tmp = g.copy()
    apply_rand_pivot(g_tmp)

    # Note the work around to not always full_reduce the graph itself!
    g_fr = g_tmp.copy()
    zx.full_reduce(g_fr)
    c_new = zx.extract_circuit(g_fr.copy()).to_basic_gates()
    c_new = zx.basic_optimization(c_new)
    if random.uniform(0, 1) < reduce_prob:
        g_tmp = g_fr.copy()
    return True, (c_new, g_tmp)

def rand_lc(c, g, reduce_prob=0.1):
    g_tmp = g.copy()
    apply_rand_lc(g_tmp)

    # Note the work around to not always full_reduce the graph itself!
    g_fr = g_tmp.copy()
    zx.full_reduce(g_fr)
    c_new = zx.extract_circuit(g_fr.copy()).to_basic_gates()
    c_new = zx.basic_optimization(c_new)
    if random.uniform(0, 1) < reduce_prob:
        g_tmp = g_fr.copy()
    return True, (c_new, g_tmp)

def do_nothing(c, g):
    return True, (c, g)

# All actions below are simplifications, not congruences
### BASIC ACTIONS

def color_change(c, g):
    g_tmp = g.copy()
    for v in g.vertices():
        if basicrules.color_change(g_tmp, v):
            try:
                c_new = zx.extract_circuit(g_tmp.copy())
                return True, (c_new, g_tmp)
            except:
                # If we can't extract the circuit, restore the graph and keep trying
                g_tmp = g.copy()
    return False, (c, g)

def strong_comp(c, g):
    g_tmp = g.copy()
    # n = g.num_vertices() # g.copy() will have consecutive vertices
    for v1 in g.vertices():
        for v2 in g.vertices():
            if basicrules.strong_comp(g_tmp, v1, v2):
                try:
                    c_new = zx.extract_circuit(g_tmp.copy())
                    return True, (c_new, g_tmp)
                except:
                    # If we can't extract the circuit, restore the graph and keep trying
                    g_tmp = g.copy()
    return False, (c, g)

def copy_X(c, g):
    g_tmp = g.copy()
    for v in g.vertices():
        if basicrules.copy_X(g_tmp, v):
            try:
                c_new = zx.extract_circuit(g_tmp.copy())
                return True, (c_new, g_tmp)
            except:
                # If we can't extract the circuit, restore the graph and keep trying
                g_tmp = g.copy()
    return False, (c, g)


def copy_Z(c, g):
    g_tmp = g.copy()
    for v in g.vertices():
        if basicrules.copy_Z(g_tmp, v):
            try:
                c_new = zx.extract_circuit(g_tmp.copy())
                return True, (c_new, g_tmp)
            except:
                # If we can't extract the circuit, restore the graph and keep trying
                g_tmp = g.copy()
    return False, (c, g)


def remove_id(c, g):
    g_tmp = g.copy()
    for v in g.vertices():
        if basicrules.remove_id(g_tmp, v):
            try:
                c_new = zx.extract_circuit(g_tmp.copy())
                return True, (c_new, g_tmp)
            except:
                # If we can't extract the circuit, restore the graph and keep trying
                g_tmp = g.copy()
    return False, (c, g)


def fuse(c, g):
    g_tmp = g.copy()
    for v1 in g.vertices():
        for v2 in g.vertices():
            if basicrules.fuse(g_tmp, v1, v2):
                try:
                    c_new = zx.extract_circuit(g_tmp.copy())
                    return True, (c_new, g_tmp)
                except:
                    # If we can't extract the circuit, restore the graph and keep trying
                    g_tmp = g.copy()
    return False, (c, g)


### INTERMEDIATE ACTIONS
# FIXME: Passes from tket and qiskit, and simplification methods from pyzx

def simp_base(c, g, simp_method):
    orig_tcount = c.tcount()
    orig_2qubitcount = c.twoqubitcount()

    g_tmp = g.copy()

    simp_method(g_tmp, quiet=True)
    try:
        # FIXME: WHY does this fail sometimes?
        c_opt = zx.extract_circuit(g_tmp.copy())
    except:
        return False, (c, g)

    opt_tcount = c_opt.tcount()
    opt_2qubitcount = c_opt.twoqubitcount()
    if orig_tcount == opt_tcount and orig_2qubitcount == opt_2qubitcount:
        return False, (c, g)
    return True, (c_opt, g_tmp)

pivot_simp = partial(simp_base, simp_method=zx.pivot_simp)
pivot_gadget_simp = partial(simp_base, simp_method=zx.pivot_gadget_simp)
pivot_boundary_simp = partial(simp_base, simp_method=zx.pivot_boundary_simp)
lcomp_simp = partial(simp_base, simp_method=zx.lcomp_simp)
bialg_simp = partial(simp_base, simp_method=zx.bialg_simp)
spider_simp = partial(simp_base, simp_method=zx.spider_simp)
id_simp = partial(simp_base, simp_method=zx.id_simp)
gadget_simp = partial(simp_base, simp_method=zx.gadget_simp)
# supplementary_simp = partial(simp_base, simp_method=zx.supplementary_simp)


# TODO: combine with transpile base
# https://qiskit.org/documentation/tutorials/circuits_advanced/04_transpiler_passes_and_passmanager.html
def qiskit_pass_base(c, g, _pass):
    orig_tcount = c.tcount()
    orig_2qubitcount = c.twoqubitcount()

    qiskit_circ = QuantumCircuit.from_qasm_str(c.to_qasm())
    pass_manager = PassManager(_pass)
    qiskit_circ_opt = pass_manager.run(qiskit_circ)
    c_opt = zx.Circuit.from_qasm(qiskit_circ_opt.qasm())

    opt_tcount = c_opt.tcount()
    opt_2qubitcount = c_opt.twoqubitcount()

    # Quick and dirty. In theory, should depend on score function
    if orig_tcount == opt_tcount and orig_2qubitcount == opt_2qubitcount:
        return False, (c, g)
    return True, (c_opt, c_opt.to_graph())

# basic_swap = partial(qiskit_pass_base, _pass=BasicSwap()) # FIXME: how to use coupling map?
cx_cancellation = partial(qiskit_pass_base, _pass=CXCancellation())
optimize_1q = partial(qiskit_pass_base, _pass=Optimize1qGates())
optimize_1q_decomp = partial(qiskit_pass_base, _pass=Optimize1qGatesDecomposition())


def tket_pass_base(c, g, _pass):
    orig_tcount = c.tcount()
    orig_2qubitcount = c.twoqubitcount()

    c_tket = pyzx_to_tk(c)
    _pass.apply(c_tket)
    RebasePyZX().apply(c_tket)
    c_opt = tk_to_pyzx(c_tket)

    opt_tcount = c_opt.tcount()
    opt_2qubitcount = c_opt.twoqubitcount()

    # Quick and dirty. In theory, should depend on score function
    if orig_tcount == opt_tcount and orig_2qubitcount == opt_2qubitcount:
        return False, (c, g)
    return True, (c_opt, c_opt.to_graph())

remove_redundancies = partial(tket_pass_base, _pass=RemoveRedundancies())
pauli_simp = partial(tket_pass_base, _pass=PauliSimp())
clifford_simp = partial(tket_pass_base, _pass=CliffordSimp())
# FIXME: Have to put into appropriate gate set first
kak_decomposition = partial(tket_pass_base, _pass=KAKDecomposition())

### ADVANCED ACTIONS

def basic_optimization(c, g):
    orig_tcount = c.tcount()
    orig_2qubitcount = c.twoqubitcount()

    c_opt = zx.basic_optimization(c)

    opt_tcount = c_opt.tcount()
    opt_2qubitcount = c_opt.twoqubitcount()
    if orig_tcount == opt_tcount and orig_2qubitcount == opt_2qubitcount:
        return False, (c, g)
    return True, (c_opt, c.to_graph())


def phase_block_optimize(c, g):
    orig_tcount = c.tcount()
    orig_2qubitcount = c.twoqubitcount()

    c_opt = zx.phase_block_optimize(c)

    opt_tcount = c_opt.tcount()
    opt_2qubitcount = c_opt.twoqubitcount()
    if orig_tcount == opt_tcount and orig_2qubitcount == opt_2qubitcount:
        return False, (c, g)
    return True, (c_opt, c.to_graph())

def full_optimize(c, g):
    orig_tcount = c.tcount()
    orig_2qubitcount = c.twoqubitcount()

    c_opt = zx.full_optimize(c)

    opt_tcount = c_opt.tcount()
    opt_2qubitcount = c_opt.twoqubitcount()
    if orig_tcount == opt_tcount and orig_2qubitcount == opt_2qubitcount:
        return False, (c, g)
    return True, (c_opt, c.to_graph())

def full_reduce(c, g):
    # FIXME: Should really be using g_tmp here?
    orig_tcount = c.tcount()
    orig_2qubitcount = c.twoqubitcount()

    zx.full_reduce(g)
    c_opt = zx.extract_circuit(g.copy())

    opt_tcount = c_opt.tcount()
    opt_2qubitcount = c_opt.twoqubitcount()
    if orig_tcount == opt_tcount and orig_2qubitcount == opt_2qubitcount:
        return False, (c, g)
    return True, (c_opt, g)

def teleport_reduce(c, g):

    orig_tcount = c.tcount()
    orig_2qubitcount = c.twoqubitcount()

    zx.teleport_reduce(g)
    try:
        c_opt = zx.Circuit.from_graph(g)
    except:
        c_opt = zx.extract_circuit(g.copy())

    opt_tcount = c_opt.tcount()
    opt_2qubitcount = c_opt.twoqubitcount()
    if orig_tcount == opt_tcount and orig_2qubitcount == opt_2qubitcount:
        return False, (c, g)
    return True, (c_opt, g)

def transpile_base(c, g, opt_level=2):
    orig_tcount = c.tcount()
    orig_2qubitcount = c.twoqubitcount()

    qiskit_circ = QuantumCircuit.from_qasm_str(c.to_qasm())
    qiskit_circ_opt = transpile(qiskit_circ, optimization_level=opt_level)
    c_opt = zx.Circuit.from_qasm(qiskit_circ_opt.qasm())

    opt_tcount = c_opt.tcount()
    opt_2qubitcount = c_opt.twoqubitcount()

    # Quick and dirty. In theory, should depend on score function
    if orig_tcount == opt_tcount and orig_2qubitcount == opt_2qubitcount:
        return False, (c, g)
    return True, (c_opt, c_opt.to_graph())

transpile0 = partial(transpile_base, opt_level=0)
transpile1 = partial(transpile_base, opt_level=1)
transpile2 = partial(transpile_base, opt_level=2)
transpile3 = partial(transpile_base, opt_level=3)


if __name__ == "__main__":
    # imports
    import matplotlib.pyplot as plt

    N_QUBITS = 5
    DEPTH = 100

    c = zx.generate.CNOT_HAD_PHASE_circuit(qubits=N_QUBITS, depth=DEPTH, clifford=False)
    g = c.to_graph()

    g_tr = g.copy()
    zx.teleport_reduce(g_tr)
    c_tr = zx.Circuit.from_graph(g_tr.copy())
    c_tr = zx.basic_optimization(c_tr).to_basic_gates()
    g_tr = c_tr.to_graph()
    to_graph_like(g_tr)

    # Testing for simplification operators
    """
    actions = [
        # BASIC
        color_change, remove_id, fuse, strong_comp, copy_X,

        # INTERMEDIATE
        spider_simp, pivot_gadget_simp, pivot_boundary_simp, lcomp_simp, bialg_simp, spider_simp,
        id_simp, gadget_simp, # supplementary_simp,
        cx_cancellation, optimize_1q, optimize_1q_decomp,
        # remove_redundancies, pauli_simp, clifford_simp, # kak_decomposition

        # ADVANCED
        teleport_reduce, transpile0, transpile1, transpile2, full_optimize, # full_reduce,
        phase_block_optimize, basic_optimization
    ]
    ga_opt = GeneticOptimizer(actions, quiet=False)
    orig_score = default_score(Mutant(c, c.to_graph()))
    print(f"Original score: {orig_score}")
    best_scores, gen_scores, c_opt = ga_opt.evolve(c, n_mutants=100, n_generations=10)


    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(list(range(len(gen_scores))), gen_scores)
    ax.axhline(orig_score, label="original", alpha=0.5, linestyle="--")
    plt.xlabel("Generation")
    plt.ylabel("Best Score in Gen")
    plt.legend()
    plt.show()
    """

    # Testing for congruence operators, using TR + Basic as input
    N_MUTANTS = 100
    N_GENS = 10
    actions = [rand_pivot, rand_lc, do_nothing]
    # actions = [rand_lc, rand_pivot]
    ga_opt = GeneticOptimizer(actions, quiet=False)
    orig_score = default_score(Mutant(c_tr, g_tr))
    print(f"Original score: {orig_score}")
    best_scores, gen_scores, c_opt = ga_opt.evolve(c_tr, n_mutants=N_MUTANTS, n_generations=N_GENS)
    reductions = [(orig_score - gen_score) / orig_score * 100 for gen_score in gen_scores]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.plot(list(range(len(scores))), scores)
    # ax.axhline(orig_score, label="original", alpha=0.5, linestyle="--")
    ax.plot(list(range(len(reductions))), reductions)
    plt.xlabel("Generation")
    plt.ylabel("Reduction")
    plt.title(f"{N_QUBITS} qubits, {DEPTH} depth, {N_MUTANTS} mutants, {N_GENS} generations")
    plt.legend()
    plt.show()
