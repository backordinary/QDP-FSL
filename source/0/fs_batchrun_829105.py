# https://github.com/suhaskvittal/foresight/blob/e68ea5184c3156eead554f4cc3e7e2091eb15fc9/src/fs_batchrun.py
"""
    author: Suhas Vittal
    date:   11 April 2022
"""

from qiskit import IBMQ

from fs_util import read_arch_file, get_error_rates_from_ibmq_backend
from fs_benchmark import benchmark_circuits
from fs_benchmark import _sabre_route, _foresight_route, _astar_route,\
                        _tket_route, _z3_route, _bip_route, _olsq_route
from fs_foresight import *
from fs_noise import google_sycamore_noise_model

IBMQ.enable_account('f0f61055f98741e1e793cc5e0dddbb89567e59362c7ec34687938a3fe50cb765d6749943e8e41ed14fe9798c1663adf7bc0cfa6389f272c54765833936e7c713')
provider = IBMQ.get_provider(hub='ibm-q-ornl', group='ornl', project='csc440')

# NOTE: Unless otherwise stated, a compiler will execute for 5 runs.
# ForeSight only needs one run as it has immediate convergence.

IBM_AUCKLAND = read_arch_file('../arch/ibm_auckland.arch')
IBM_MANILA = read_arch_file('../arch/ibm_manila.arch')
IBM_TOKYO = read_arch_file('../arch/ibm_tokyo.arch')
GOOGLE_SYCAMORE = read_arch_file('../arch/google_weber.arch')
RIGETTI_ASPEN9 = read_arch_file('../arch/rigetti_aspen9.arch')
IBM_TORONTO = read_arch_file('../arch/ibm_toronto.arch')
IBM_HANOI = read_arch_file('../arch/ibm_hanoi.arch')
IBM_MONTREAL = read_arch_file('../arch/ibmq_montreal.arch')
IBM_KOLKATA = read_arch_file('../arch/ibmq_kolkata.arch')
IBM_MUMBAI = read_arch_file('../arch/ibmq_mumbai.arch')
IBM_HEAVYHEX = read_arch_file('../arch/ibm_3heavyhex.arch')
GRID100 = read_arch_file('../arch/100grid.arch')
GRID500 = read_arch_file('../arch/500grid.arch')

# MAIN BENCHMARKS

def batch001():
    foresight = ForeSight(
        GOOGLE_SYCAMORE,
        slack=2,
        solution_cap=64,
        flags=FLAG_ALAP
    )
    _foresight_alap = lambda x,y: _foresight_route(x,y,foresight)

    benchmark_circuits(
        '../benchmarks/mapped_circuits/google_sycamore',
        '../arch/google_weber.arch',
        'foresight_alap',
        _foresight_alap,
        runs=1
    )

def batch002():
    foresight = ForeSight(
        RIGETTI_ASPEN9,
        slack=2,
        solution_cap=64,
        flags=FLAG_ALAP
    )
    _foresight_alap = lambda x,y: _foresight_route(x,y,foresight)

    benchmark_circuits(
        '../benchmarks/mapped_circuits/rigetti_aspen9',
        '../arch/rigetti_aspen9.arch',
        'foresight_alap',
        _foresight_alap,
        runs=1,
    )

def batch003():
    foresight = ForeSight(
        IBM_TOKYO,
        slack=2,
        solution_cap=64,
        flags=FLAG_ALAP
    )
    _foresight_alap = lambda x,y: _foresight_route(x,y,foresight)

    benchmark_circuits(
        '../benchmarks/mapped_circuits/ibm_tokyo',
        '../arch/ibm_tokyo.arch',
        'foresight_alap',
        _foresight_alap,
        runs=1
    )

def batch004():
    foresight = ForeSight(
        IBM_TORONTO,
        slack=1,
        solution_cap=64,
        flags=FLAG_ALAP
    )
    _foresight_alap = lambda x,y: _foresight_route(x,y,foresight)

    benchmark_circuits(
        '../benchmarks/mapped_circuits/ibm_toronto',
        '../arch/ibm_toronto.arch',
        'foresight_alap',
        _foresight_alap,
        runs=1
    )

def batch005():
    foresight = ForeSight(
        IBM_HEAVYHEX,
        slack=1,
        solution_cap=64,
        flags=FLAG_ALAP
    )
    _foresight_alap = lambda x,y: _foresight_route(x,y,foresight)

    benchmark_circuits(
        '../benchmarks/mapped_circuits/ibm_heavyhex',
        '../arch/ibm_3heavyhex.arch',
        'foresight_alap',
        _foresight_alap,
        runs=1
    )

def batch006():
    foresight = ForeSight(
        GOOGLE_SYCAMORE,
        slack=2,
        solution_cap=64,
        flags=FLAG_ASAP
    )
    _foresight_asap = lambda x,y: _foresight_route(x,y,foresight)

    benchmark_circuits(
        '../benchmarks/mapped_circuits/google_sycamore',
        '../arch/google_weber.arch',
        'foresight_asap',
        _foresight_asap,
        runs=1
    )

def batch007():
    foresight = ForeSight(
        RIGETTI_ASPEN9,
        slack=2,
        solution_cap=64,
        flags=FLAG_ASAP
    )
    _foresight_asap = lambda x,y: _foresight_route(x,y,foresight)

    benchmark_circuits(
        '../benchmarks/mapped_circuits/rigetti_aspen9',
        '../arch/rigetti_aspen9.arch',
        'foresight_asap',
        _foresight_asap,
        runs=1,
    )

def batch008():
    foresight = ForeSight(
        IBM_TOKYO,
        slack=2,
        solution_cap=64,
        flags=FLAG_ASAP
    )
    _foresight_asap = lambda x,y: _foresight_route(x,y,foresight)

    benchmark_circuits(
        '../benchmarks/mapped_circuits/ibm_tokyo',
        '../arch/ibm_tokyo.arch',
        'foresight_asap',
        _foresight_asap,
        runs=1
    )

def batch009():
    foresight = ForeSight(
        IBM_TORONTO,
        slack=1,
        solution_cap=64,
        flags=FLAG_ASAP
    )
    _foresight_asap = lambda x,y: _foresight_route(x,y,foresight)

    benchmark_circuits(
        '../benchmarks/mapped_circuits/ibm_toronto',
        '../arch/ibm_toronto.arch',
        'foresight_asap',
        _foresight_asap,
        runs=1
    )

def batch010():
    foresight = ForeSight(
        IBM_HEAVYHEX,
        slack=1,
        solution_cap=64,
        flags=FLAG_ASAP
    )
    _foresight_asap = lambda x,y: _foresight_route(x,y,foresight)

    benchmark_circuits(
        '../benchmarks/mapped_circuits/ibm_heavyhex',
        '../arch/ibm_3heavyhex.arch',
        'foresight_asap',
        _foresight_asap,
        runs=1
    )

def batch011():
    benchmark_circuits(
        '../benchmarks/mapped_circuits/ibm_tokyo',
        '../arch/ibm_tokyo.arch',
        'sabre',
        _sabre_route
    )

def batch012():
    benchmark_circuits(
        '../benchmarks/mapped_circuits/google_sycamore',
        '../arch/google_weber.arch',
        'sabre',
        _sabre_route
    )

def batch013():
    benchmark_circuits(
        '../benchmarks/mapped_circuits/rigetti_aspen9',
        '../arch/rigetti_aspen9.arch',
        'sabre',
        _sabre_route
    )

def batch014():
    benchmark_circuits(
        '../benchmarks/mapped_circuits/ibm_toronto',
        '../arch/ibm_toronto.arch',
        'sabre',
        _sabre_route
    )

def batch015():
    benchmark_circuits(
        '../benchmarks/mapped_circuits/ibm_heavyhex',
        '../arch/ibm_3heavyhex.arch',
        'sabre',
        _sabre_route
    )

def batch016():
    benchmark_circuits(
        '../benchmarks/mapped_circuits/ibm_tokyo',
        '../arch/ibm_tokyo.arch',
        'astar',
        _astar_route
    )

def batch017():
    benchmark_circuits(
        '../benchmarks/mapped_circuits/google_sycamore',
        '../arch/google_weber.arch',
        'astar',
        _astar_route
    )

def batch018():
    benchmark_circuits(
        '../benchmarks/mapped_circuits/rigetti_aspen9',
        '../arch/rigetti_aspen9.arch',
        'astar',
        _astar_route
    )

def batch019():
    benchmark_circuits(
        '../benchmarks/mapped_circuits/ibm_toronto',
        '../arch/ibm_toronto.arch',
        'astar',
        _astar_route
    )

def batch020():
    benchmark_circuits(
        '../benchmarks/mapped_circuits/ibm_heavyhex',
        '../arch/ibm_3heavyhex.arch',
        'astar',
        _astar_route
    )

def batch021():
    benchmark_circuits(
        '../benchmarks/mapped_circuits/ibm_tokyo',
        '../arch/ibm_tokyo.arch',
        'tket',
        _tket_route
    )

def batch022():
    benchmark_circuits(
        '../benchmarks/mapped_circuits/google_sycamore',
        '../arch/google_weber.arch',
        'tket',
        _tket_route
    )

def batch023():
    benchmark_circuits(
        '../benchmarks/mapped_circuits/rigetti_aspen9',
        '../arch/rigetti_aspen9.arch',
        'tket',
        _tket_route
    )

def batch024():
    benchmark_circuits(
        '../benchmarks/mapped_circuits/ibm_toronto',
        '../arch/ibm_toronto.arch',
        'tket',
        _tket_route
    )

def batch025():
    benchmark_circuits(
        '../benchmarks/mapped_circuits/ibm_heavyhex',
        '../arch/ibm_3heavyhex.arch',
        'tket',
        _tket_route
    )


# GENERAL SENSITIVITY ANALYSIS ON GOOGLE SYCAMORE USING REVLIB

def batch101():  
    def genforesight(slack, solution_cap):
        foresight = ForeSight(
            GOOGLE_SYCAMORE,
            slack=slack,
            solution_cap=solution_cap,
            flags=FLAG_ALAP
        )
        return (lambda x,y: _foresight_route(x,y,foresight))
    foresight_table = {}
    for slack in [0,1,2,3,4]:
        for solution_cap in [4,8,16,32,64,128]:
            _foresight = genforesight(slack,solution_cap)
            foresight_table[(slack,solution_cap)] = _foresight
    benchmark_circuits(
        '../benchmarks/sensitivity/gensens_sycamore',
        '../arch/google_weber.arch',
        'sabre',
        _sabre_route
    )
    for solution_cap in [4,8,16,32,64,128]:
        _foresight = foresight_table[(2,solution_cap)]
        benchmark_circuits(
            '../benchmarks/sensitivity/gensens_sycamore',
            '../arch/google_weber.arch',
            'fs_%d_%d' % (2, solution_cap),
            _foresight,
            runs=1,
            memory=False
        )

# TIME AND MEMORY COMPLEXITY ANALYSIS USING BV50 AND BV100 ON 100 QUBIT GRID

def batch201():
    def genforesight(slack, solution_cap):
        foresight = ForeSight(
            GRID100,
            slack=slack,
            solution_cap=solution_cap,
            flags=FLAG_ALAP
        )
        return (lambda x,y: _foresight_route(x,y,foresight))
    foresight_table = {}
    for slack in [0,1,2,3,4]:
        for solution_cap in [4,8,16,32,64,128]:
            _foresight = genforesight(slack,solution_cap)
            foresight_table[(slack,solution_cap)] = _foresight
    for slack in [0,1,2,3,4]:
        for solution_cap in [4,8,16,32,64,128]:
            print('slack = %d, solution_cap=%d' % (slack, solution_cap))
            _foresight = foresight_table[(slack,solution_cap)]
            benchmark_circuits(
                '../benchmarks/sensitivity/tmsens_100grid',
                '../arch/100grid.arch',
                'fs_%d_%d' % (slack, solution_cap),
                _foresight,
                runs=1
            )

# SENSITIVITY ANALYSIS ON BV100, BV200, BV500 ON 500 QUBIT GRID

def batch301():
    foresight = ForeSight(
        GRID500,
        slack=2,
        solution_cap=64,
        flags=FLAG_ALAP
    )
    _foresight = lambda x,y: _foresight_route(x,y,foresight)
    benchmark_circuits(
        '../benchmarks/sensitivity/bvsens_500grid',
        '../arch/500grid.arch',
        'sabre',
        _sabre_route
    )
    benchmark_circuits(
        '../benchmarks/sensitivity/bvsens_500grid',
        '../arch/500grid.arch',
        'foresight',
        _foresight,
        runs=1
    )

# NOISE AWARE ROUTING

def batch401():
    _, cx_error_rates, sq_error_rates, ro_error_rates =\
        google_sycamore_noise_model(GOOGLE_SYCAMORE, '../arch/noisy/google_weber.noise')
    cx_error_rate_list = [cx_error_rates[c] for c in cx_error_rates]
    # Compute statistics
    mean_cx_error_rate = np.mean(cx_error_rate_list)
    min_cx_error_rate = np.min(cx_error_rate_list)
    max_cx_error_rate = np.max(cx_error_rate_list)
    # We have found that ALAP ForeSight performs best with delta=mean, ASAP is best with delta=max.
    foresight_noise_unaware_alap = ForeSight(
        GOOGLE_SYCAMORE,
        slack=2,
        solution_cap=64,
        flags=FLAG_ALAP
    )
    foresight_noise_aware_alap = ForeSight(
        GOOGLE_SYCAMORE,
        slack=mean_cx_error_rate,
        solution_cap=64,
        cx_error_rates=cx_error_rates,
        sq_error_rates=sq_error_rates,
        ro_error_rates=ro_error_rates,
        flags=FLAG_ALAP|FLAG_NOISE_AWARE
    )
    foresight_noise_unaware_asap = ForeSight(
        GOOGLE_SYCAMORE,
        slack=2,
        solution_cap=64,
        flags=FLAG_ASAP
    )
    foresight_noise_aware_asap = ForeSight(
        GOOGLE_SYCAMORE,
        slack=max_cx_error_rate,
        solution_cap=64,
        cx_error_rates=cx_error_rates,
        sq_error_rates=sq_error_rates,
        ro_error_rates=ro_error_rates,
        flags=FLAG_ASAP|FLAG_NOISE_AWARE
    )
    _fs1 = lambda x,y: _foresight_route(x,y,foresight_noise_unaware_alap)
    _fs2 = lambda x,y: _foresight_route(x,y,foresight_noise_aware_alap)
    _fs3 = lambda x,y: _foresight_route(x,y,foresight_noise_unaware_asap)
    _fs4 = lambda x,y: _foresight_route(x,y,foresight_noise_aware_asap)
    benchmark_circuits(
        '../benchmarks/fidelity_tests/noise_sycamore',
        '../arch/google_weber.arch',
        'sabre',
        _sabre_route
    )
    benchmark_circuits(
        '../benchmarks/fidelity_tests/noise_sycamore',
        '../arch/google_weber.arch',
        'foresight_alap',
        _fs1,
        runs=1
    )
    benchmark_circuits(
        '../benchmarks/fidelity_tests/noise_sycamore',
        '../arch/google_weber.arch',
        'noisy_foresight_alap',
        _fs2,
        runs=1
    )
    benchmark_circuits(
        '../benchmarks/fidelity_tests/noise_sycamore',
        '../arch/google_weber.arch',
        'foresight_asap',
        _fs1,
        runs=1
    )
    benchmark_circuits(
        '../benchmarks/fidelity_tests/noise_sycamore',
        '../arch/google_weber.arch',
        'noisy_foresight_asap',
        _fs2,
        runs=1
    )

def batch402():
    hanoi_backend = provider.get_backend('ibm_hanoi')
    sq_error_rates, cx_error_rates, ro_error_rates, mean_coh_t1, mean_cx_time =\
        get_error_rates_from_ibmq_backend(hanoi_backend)
    cx_error_rate_list = [cx_error_rates[c] for c in cx_error_rates]
    # Compute statistics
    mean_cx_error_rate = np.mean(cx_error_rate_list)
    min_cx_error_rate = np.min(cx_error_rate_list)
    max_cx_error_rate = np.max(cx_error_rate_list)

    # We have found that ALAP ForeSight performs best with delta=mean, ASAP is best with delta=max.
    foresight_noise_unaware_alap = ForeSight(
        IBM_HANOI,
        slack=1,
        solution_cap=64,
        flags=FLAG_ALAP
    )
    foresight_noise_aware_alap = ForeSight(
        IBM_HANOI,
        slack=mean_cx_error_rate,
        solution_cap=64,
        cx_error_rates=cx_error_rates,
        sq_error_rates=sq_error_rates,
        ro_error_rates=ro_error_rates,
        coh_t1=mean_coh_t1,
        layer_time=mean_cx_time,
        flags=FLAG_ALAP|FLAG_NOISE_AWARE
    )
    foresight_noise_unaware_asap = ForeSight(
        IBM_HANOI,
        slack=1,
        solution_cap=64,
        flags=FLAG_ASAP
    )
    foresight_noise_aware_asap = ForeSight(
        IBM_HANOI,
        slack=max_cx_error_rate,
        solution_cap=64,
        cx_error_rates=cx_error_rates,
        sq_error_rates=sq_error_rates,
        ro_error_rates=ro_error_rates,
        coh_t1=mean_coh_t1,
        layer_time=mean_cx_time,
        flags=FLAG_ASAP|FLAG_NOISE_AWARE
    )
    _fs1 = lambda x,y: _foresight_route(x,y,foresight_noise_unaware_alap)
    _fs2 = lambda x,y: _foresight_route(x,y,foresight_noise_aware_alap)
    _fs3 = lambda x,y: _foresight_route(x,y,foresight_noise_unaware_asap)
    _fs4 = lambda x,y: _foresight_route(x,y,foresight_noise_aware_asap)
    benchmark_circuits(
        '../benchmarks/fidelity_tests/ibm_hanoi',
        '../arch/ibm_hanoi.arch',
        'sabre',
        _sabre_route
    )
    benchmark_circuits(
        '../benchmarks/fidelity_tests/ibm_hanoi',
        '../arch/ibm_hanoi.arch',
        'foresight_alap',
        _fs1,
        runs=1
    )
    benchmark_circuits(
        '../benchmarks/fidelity_tests/ibm_hanoi',
        '../arch/ibm_hanoi.arch',
        'noisy_foresight_alap',
        _fs2,
        runs=1
    )
    benchmark_circuits(
        '../benchmarks/fidelity_tests/ibm_hanoi',
        '../arch/ibm_hanoi.arch',
        'foresight_asap',
        _fs1,
        runs=1
    )
    benchmark_circuits(
        '../benchmarks/fidelity_tests/ibm_hanoi',
        '../arch/ibm_hanoi.arch',
        'noisy_foresight_asap',
        _fs2,
        runs=1
    )

def batch403():
    ibmq_backend = provider.get_backend('ibmq_montreal')
    sq_error_rates, cx_error_rates, ro_error_rates, mean_coh_t1, mean_cx_time =\
        get_error_rates_from_ibmq_backend(ibmq_backend)
    cx_error_rate_list = [cx_error_rates[c] for c in cx_error_rates]
    # Compute statistics
    mean_cx_error_rate = np.mean(cx_error_rate_list)
    min_cx_error_rate = np.min(cx_error_rate_list)
    max_cx_error_rate = np.max(cx_error_rate_list)

    # We have found that ALAP ForeSight performs best with delta=mean, ASAP is best with delta=max.
    foresight_noise_unaware_alap = ForeSight(
        IBM_MONTREAL,
        slack=1,
        solution_cap=64,
        flags=FLAG_ALAP
    )
    foresight_noise_aware_alap = ForeSight(
        IBM_MONTREAL,
        slack=mean_cx_error_rate,
        solution_cap=64,
        cx_error_rates=cx_error_rates,
        sq_error_rates=sq_error_rates,
        ro_error_rates=ro_error_rates,
        coh_t1=mean_coh_t1,
        layer_time=mean_cx_time,
        flags=FLAG_ALAP|FLAG_NOISE_AWARE
    )
    foresight_noise_unaware_asap = ForeSight(
        IBM_MONTREAL,
        slack=1,
        solution_cap=64,
        flags=FLAG_ASAP
    )
    foresight_noise_aware_asap = ForeSight(
        IBM_MONTREAL,
        slack=max_cx_error_rate,
        solution_cap=64,
        cx_error_rates=cx_error_rates,
        sq_error_rates=sq_error_rates,
        ro_error_rates=ro_error_rates,
        coh_t1=mean_coh_t1,
        layer_time=mean_cx_time,
        flags=FLAG_ASAP|FLAG_NOISE_AWARE
    )
    _fs1 = lambda x,y: _foresight_route(x,y,foresight_noise_unaware_alap)
    _fs2 = lambda x,y: _foresight_route(x,y,foresight_noise_aware_alap)
    _fs3 = lambda x,y: _foresight_route(x,y,foresight_noise_unaware_asap)
    _fs4 = lambda x,y: _foresight_route(x,y,foresight_noise_aware_asap)
    benchmark_circuits(
        '../benchmarks/fidelity_tests/ibmq_montreal',
        '../arch/ibmq_montreal.arch',
        'sabre',
        _sabre_route
    )
    benchmark_circuits(
        '../benchmarks/fidelity_tests/ibmq_montreal',
        '../arch/ibmq_montreal.arch',
        'foresight_alap',
        _fs1,
        runs=1
    )
    benchmark_circuits(
        '../benchmarks/fidelity_tests/ibmq_montreal',
        '../arch/ibmq_montreal.arch',
        'noisy_foresight_alap',
        _fs2,
        runs=1
    )
    benchmark_circuits(
        '../benchmarks/fidelity_tests/ibmq_montreal',
        '../arch/ibmq_montreal.arch',
        'foresight_asap',
        _fs1,
        runs=1
    )
    benchmark_circuits(
        '../benchmarks/fidelity_tests/ibmq_montreal',
        '../arch/ibmq_montreal.arch',
        'noisy_foresight_asap',
        _fs2,
        runs=1
    )

def batch404():
    ibmq_backend = provider.get_backend('ibmq_mumbai')
    sq_error_rates, cx_error_rates, ro_error_rates, mean_coh_t1, mean_cx_time =\
        get_error_rates_from_ibmq_backend(ibmq_backend)
    cx_error_rate_list = [cx_error_rates[c] for c in cx_error_rates]
    # Compute statistics
    mean_cx_error_rate = np.mean(cx_error_rate_list)
    min_cx_error_rate = np.min(cx_error_rate_list)
    max_cx_error_rate = np.max(cx_error_rate_list)

    # We have found that ALAP ForeSight performs best with delta=mean, ASAP is best with delta=max.
    foresight_noise_unaware_alap = ForeSight(
        IBM_MUMBAI,
        slack=1,
        solution_cap=64,
        flags=FLAG_ALAP
    )
    foresight_noise_aware_alap = ForeSight(
        IBM_MUMBAI,
        slack=mean_cx_error_rate,
        solution_cap=64,
        cx_error_rates=cx_error_rates,
        sq_error_rates=sq_error_rates,
        ro_error_rates=ro_error_rates,
        coh_t1=mean_coh_t1,
        layer_time=mean_cx_time,
        flags=FLAG_ALAP|FLAG_NOISE_AWARE
    )
    foresight_noise_unaware_asap = ForeSight(
        IBM_MUMBAI,
        slack=1,
        solution_cap=64,
        flags=FLAG_ASAP
    )
    foresight_noise_aware_asap = ForeSight(
        IBM_MUMBAI,
        slack=max_cx_error_rate,
        solution_cap=64,
        cx_error_rates=cx_error_rates,
        sq_error_rates=sq_error_rates,
        ro_error_rates=ro_error_rates,
        coh_t1=mean_coh_t1,
        layer_time=mean_cx_time,
        flags=FLAG_ASAP|FLAG_NOISE_AWARE
    )
    _fs1 = lambda x,y: _foresight_route(x,y,foresight_noise_unaware_alap)
    _fs2 = lambda x,y: _foresight_route(x,y,foresight_noise_aware_alap)
    _fs3 = lambda x,y: _foresight_route(x,y,foresight_noise_unaware_asap)
    _fs4 = lambda x,y: _foresight_route(x,y,foresight_noise_aware_asap)
    benchmark_circuits(
        '../benchmarks/fidelity_tests/ibmq_mumbai',
        '../arch/ibmq_mumbai.arch',
        'sabre',
        _sabre_route
    )
    benchmark_circuits(
        '../benchmarks/fidelity_tests/ibmq_mumbai',
        '../arch/ibmq_mumbai.arch',
        'foresight_alap',
        _fs1,
        runs=1
    )
    benchmark_circuits(
        '../benchmarks/fidelity_tests/ibmq_mumbai',
        '../arch/ibmq_mumbai.arch',
        'noisy_foresight_alap',
        _fs2,
        runs=1
    )
    benchmark_circuits(
        '../benchmarks/fidelity_tests/ibmq_mumbai',
        '../arch/ibmq_mumbai.arch',
        'foresight_asap',
        _fs1,
        runs=1
    )
    benchmark_circuits(
        '../benchmarks/fidelity_tests/ibmq_mumbai',
        '../arch/ibmq_mumbai.arch',
        'noisy_foresight_asap',
        _fs2,
        runs=1
    )

def batch405():
    ibmq_backend = provider.get_backend('ibm_auckland')
    sq_error_rates, cx_error_rates, ro_error_rates, mean_coh_t1, mean_cx_time =\
        get_error_rates_from_ibmq_backend(ibmq_backend)
    cx_error_rate_list = [cx_error_rates[c] for c in cx_error_rates]
    # Compute statistics
    mean_cx_error_rate = np.mean(cx_error_rate_list)
    min_cx_error_rate = np.min(cx_error_rate_list)
    max_cx_error_rate = np.max(cx_error_rate_list)

    # We have found that ALAP ForeSight performs best with delta=mean, ASAP is best with delta=max.
    foresight_noise_unaware_alap = ForeSight(
        IBM_AUCKLAND,
        slack=1,
        solution_cap=64,
        flags=FLAG_ALAP
    )
    foresight_noise_aware_alap = ForeSight(
        IBM_AUCKLAND,
        slack=mean_cx_error_rate,
        solution_cap=64,
        cx_error_rates=cx_error_rates,
        sq_error_rates=sq_error_rates,
        ro_error_rates=ro_error_rates,
        coh_t1=mean_coh_t1,
        layer_time=mean_cx_time,
        flags=FLAG_ALAP|FLAG_NOISE_AWARE
    )
    foresight_noise_unaware_asap = ForeSight(
        IBM_AUCKLAND,
        slack=1,
        solution_cap=64,
        flags=FLAG_ASAP
    )
    foresight_noise_aware_asap = ForeSight(
        IBM_AUCKLAND,
        slack=max_cx_error_rate,
        solution_cap=64,
        cx_error_rates=cx_error_rates,
        sq_error_rates=sq_error_rates,
        ro_error_rates=ro_error_rates,
        coh_t1=mean_coh_t1,
        layer_time=mean_cx_time,
        flags=FLAG_ASAP|FLAG_NOISE_AWARE
    )
    _fs1 = lambda x,y: _foresight_route(x,y,foresight_noise_unaware_alap)
    _fs2 = lambda x,y: _foresight_route(x,y,foresight_noise_aware_alap)
    _fs3 = lambda x,y: _foresight_route(x,y,foresight_noise_unaware_asap)
    _fs4 = lambda x,y: _foresight_route(x,y,foresight_noise_aware_asap)
    benchmark_circuits(
        '../benchmarks/fidelity_tests/ibm_auckland',
        '../arch/ibm_auckland.arch',
        'sabre',
        _sabre_route
    )
    benchmark_circuits(
        '../benchmarks/fidelity_tests/ibm_auckland',
        '../arch/ibm_auckland.arch',
        'foresight_alap',
        _fs1,
        runs=1
    )
    benchmark_circuits(
        '../benchmarks/fidelity_tests/ibm_auckland',
        '../arch/ibm_auckland.arch',
        'noisy_foresight_alap',
        _fs2,
        runs=1
    )
    benchmark_circuits(
        '../benchmarks/fidelity_tests/ibm_auckland',
        '../arch/ibm_auckland.arch',
        'foresight_asap',
        _fs1,
        runs=1
    )
    benchmark_circuits(
        '../benchmarks/fidelity_tests/ibm_auckland',
        '../arch/ibm_auckland.arch',
        'noisy_foresight_asap',
        _fs2,
        runs=1
    )

def batch406():
    ibmq_backend = provider.get_backend('ibmq_kolkata')
    sq_error_rates, cx_error_rates, ro_error_rates, mean_coh_t1, mean_cx_time =\
        get_error_rates_from_ibmq_backend(ibmq_backend)
    cx_error_rate_list = [cx_error_rates[c] for c in cx_error_rates]
    # Compute statistics
    mean_cx_error_rate = np.mean(cx_error_rate_list)
    min_cx_error_rate = np.min(cx_error_rate_list)
    max_cx_error_rate = np.max(cx_error_rate_list)

    # We have found that ALAP ForeSight performs best with delta=mean, ASAP is best with delta=max.
    foresight_noise_unaware_alap = ForeSight(
        IBM_KOLKATA,
        slack=1,
        solution_cap=64,
        flags=FLAG_ALAP
    )
    foresight_noise_aware_alap = ForeSight(
        IBM_KOLKATA,
        slack=mean_cx_error_rate,
        solution_cap=64,
        cx_error_rates=cx_error_rates,
        sq_error_rates=sq_error_rates,
        ro_error_rates=ro_error_rates,
        coh_t1=mean_coh_t1,
        layer_time=mean_cx_time,
        flags=FLAG_ALAP|FLAG_NOISE_AWARE
    )
    foresight_noise_unaware_asap = ForeSight(
        IBM_KOLKATA,
        slack=1,
        solution_cap=64,
        flags=FLAG_ASAP
    )
    foresight_noise_aware_asap = ForeSight(
        IBM_KOLKATA,
        slack=max_cx_error_rate,
        solution_cap=64,
        cx_error_rates=cx_error_rates,
        sq_error_rates=sq_error_rates,
        ro_error_rates=ro_error_rates,
        coh_t1=mean_coh_t1,
        layer_time=mean_cx_time,
        flags=FLAG_ASAP|FLAG_NOISE_AWARE
    )
    _fs1 = lambda x,y: _foresight_route(x,y,foresight_noise_unaware_alap)
    _fs2 = lambda x,y: _foresight_route(x,y,foresight_noise_aware_alap)
    _fs3 = lambda x,y: _foresight_route(x,y,foresight_noise_unaware_asap)
    _fs4 = lambda x,y: _foresight_route(x,y,foresight_noise_aware_asap)
    benchmark_circuits(
        '../benchmarks/fidelity_tests/ibmq_kolkata',
        '../arch/ibmq_kolkata.arch',
        'sabre',
        _sabre_route
    )
    benchmark_circuits(
        '../benchmarks/fidelity_tests/ibmq_kolkata',
        '../arch/ibmq_kolkata.arch',
        'foresight_alap',
        _fs1,
        runs=1
    )
    benchmark_circuits(
        '../benchmarks/fidelity_tests/ibmq_kolkata',
        '../arch/ibmq_kolkata.arch',
        'noisy_foresight_alap',
        _fs2,
        runs=1
    )
    benchmark_circuits(
        '../benchmarks/fidelity_tests/ibmq_kolkata',
        '../arch/ibmq_kolkata.arch',
        'foresight_asap',
        _fs1,
        runs=1
    )
    benchmark_circuits(
        '../benchmarks/fidelity_tests/ibmq_kolkata',
        '../arch/ibmq_kolkata.arch',
        'noisy_foresight_asap',
        _fs2,
        runs=1
    )

# SOLVER COMPARISION (Z3) VS FORESIGHT AND SABRE

def batch501():
    foresight_alap = ForeSight(
        IBM_MANILA,
        slack=2,
        solution_cap=64,
        flags=FLAG_ALAP
    )
    _foresight_alap = lambda x,y: _foresight_route(x,y,foresight_alap)

    benchmark_circuits(
        '../benchmarks/solver_circuits/ibm_manila',
        '../arch/ibm_manila.arch',
        'foresight_alap',
        _foresight_alap,
        runs=1
    )

def batch502():
    foresight_asap = ForeSight(
        IBM_MANILA,
        slack=2,
        solution_cap=64,
        flags=FLAG_ALAP
    )
    _foresight_asap = lambda x,y: _foresight_route(x,y,foresight_asap)

    benchmark_circuits(
        '../benchmarks/solver_circuits/ibm_manila',
        '../arch/ibm_manila.arch',
        'foresight_asap',
        _foresight_asap,
        runs=1
    )

def batch503():
    benchmark_circuits(
        '../benchmarks/solver_circuits/ibm_manila',
        '../arch/ibm_manila.arch',
        'sabre',
        _sabre_route
    )

def batch504():
    benchmark_circuits(
        '../benchmarks/solver_circuits/ibm_manila',
        '../arch/ibm_manila.arch',
        'z3solver',
        _z3_route
    )

def batch505():
    benchmark_circuits(
        '../benchmarks/solver_circuits/ibm_manila',
        '../arch/ibm_manila.arch',
        'bipsolver',
        _bip_route
    )

def batch506():
    benchmark_circuits(
        '../benchmarks/solver_circuits/ibm_manila',
        '../arch/ibm_manila.arch',
        'olsq',
        _olsq_route
    )

if __name__ == '__main__':
    from sys import argv
    batch_no = argv[1]
    exec('batch%s()' % batch_no)

