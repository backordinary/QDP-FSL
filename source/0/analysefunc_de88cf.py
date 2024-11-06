# https://github.com/jarndejong/FTSWAP_experiment/blob/b9d9d171ea25d712d3f77285119c490b018e46e0/filesfortijn/Analysis/Analysefunc.py
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 15:22:48 2018

@author: Jarnd
"""
import qiskit.tools.qcvv.tomography as tomo
import Analysis.tomography_functions as tomoself
import numpy as np

#%% Fitting of tomodata to chi
def fit_tomodata_multiple(meas_data_all, tomo_set, B_chi, B_choi, n,stddv=True):
    '''
    Fit the measurement tomography data from multiple datasets in meas_data_all specified by tomo_set to a chi and a corresponding choi matrix.
    There are 3 different methods:
        -'own' (standard): This method used the _fit_tomodata_own_ function, which is specified in that docstring.
            This method uses the _fit_tomodata_own() function. It provides a CP and TP chi and choi matrix.
            Furthermore, it calculates the standard deviation on chi (and choi) if stddv==True (standard)
            For details, see the docstring of that function.
        
    The other two methods use the interal qiskit function qiskit.tools.qcvv.tomography.fit_tomography_data()
    Both these methods use the wrapper function _fit_tomodata_qiskit_(). For details, see that docstring.
        -'wizard' : This method gives a CP choi and chi matrix but has problems with trace preservation.
        -'leastsq': This method is a straightforward least squares fit that does not promise CP, but has less problems with TP.
        
    The process matrices are returned as:
        ((chi,chi_stddv),(choi,choi_stddv))
    '''
    B_chi = tomoself.get_pauli_basis(n, normalise=False)                                         # Get the basis in which chi is expressed
    B_prep = tomoself.get_pauli_basis(n, normalise=False)                       # Get the basis in which the experiments are prepared
    B_meas = tomoself.get_pauli_basis(n, normalise=False)                       # Get the basis in which the measurements are done
    
    lam, lampau, lamstddv = tomoself.get_lambda_from_meas_multiple(tomo_set,             # Get the vectors lambda and lambda_stddv from the tomography data
                                                          meas_data_all, n)
    A = tomoself.get_A_mat(B_prep, B_meas, B_chi)                               # Calculate the A matrix from the prep, meas and chi basis
    chivect = np.linalg.solve(A, lam)                                           # Invert to calculate chi in vectorform
    Ainv = np.linalg.inv(A)                                                     # Calculate the inverse of A for error calculations (A is full rank)
    Ainvsq = np.abs(Ainv)*np.abs(Ainv);                                         # Calculate the elementwise square of Ainv
    lamstddvsq = lamstddv*lamstddv;                                             # Calculate the elementwise square of l_stddv
    chistddvvect = np.sqrt(Ainvsq @ lamstddvsq)                                 # Calculate the standard deviation on chi using the method from the description
    chi = np.reshape(chivect, ((2*n)**2, (2*n)**2))                             # Reshape into a square matrix
    print('Minimum eigenvalue before: ', np.min(np.linalg.eigvals(chi)))
    chi_stddv = np.reshape(chistddvvect, ((2*n)**2, (2*n)**2))                   # Reshape into a square matrix
    print('largest eigenvalue before: ',np.max(np.linalg.eigvals(chi)))
    num = np.max(np.linalg.eigvals(chi)) + np.abs(np.min(np.linalg.eigvals(chi)))
    den = 1+16*np.abs(np.min(np.linalg.eigvals(chi)))
    print('test: ',num/den,'times 4:',4*num/den)
    #chi = make_CP(chi,n)
    print('Minimum eigenvalue after: ', np.min(np.linalg.eigvals(chi)))
    print('largest eigenvalue after: ',np.max(np.linalg.eigvals(chi)))
    choi = tomoself.chi_to_choi(chi,B_choi, n)
    choi_stddv = tomoself.chi_to_choi(chi_stddv, B_choi, n)
    return ((chi,chi_stddv),(choi,choi_stddv))

def fit_tomodata(tomo_data, tomo_set, B_chi, B_choi, n, method='own',stddv=True):
    '''
    Fit the tomography data from tomo_data specified by tomo_set to a chi and a corresponding choi matrix.
    There are 3 different methods:
        -'own' (standard): This method used the _fit_tomodata_own_ function, which is specified in that docstring.
            This method uses the _fit_tomodata_own() function. It provides a CP and TP chi and choi matrix.
            Furthermore, it calculates the standard deviation on chi (and choi) if stddv==True (standard)
            For details, see the docstring of that function.
        
    The other two methods use the interal qiskit function qiskit.tools.qcvv.tomography.fit_tomography_data()
    Both these methods use the wrapper function _fit_tomodata_qiskit_(). For details, see that docstring.
        -'wizard' : This method gives a CP choi and chi matrix but has problems with trace preservation.
        -'leastsq': This method is a straightforward least squares fit that does not promise CP, but has less problems with TP.
        
    The process matrices are returned as:
        ((chi,chi_stddv),(choi,choi_stddv))
    '''
    if method == 'own':
        chi, chi_stddv = _fit_tomodata_own_(tomo_data, tomo_set, n, stddv)
        chi = make_CP(chi,n)
        choi = tomoself.chi_to_choi(chi,B_choi, n)
        choi_stddv = tomoself.chi_to_choi(chi_stddv, B_choi, n)
    elif method == 'wizard':
        choi = _fit_tomodata_qiskit_(tomo_data, method='wizard')
        chi = tomoself.choi_to_chi(choi, B_choi, n)
        chi_stddv = np.zeros_like(chi)
        choi_stddv = np.zeros_like(chi)
        print('Warning: no standard deviation calculated!')
    elif method == 'leastsq':
        choi = _fit_tomodata_qiskit_(tomo_data, method='leastsq')
        choi = make_CP(choi,n)
        chi = tomoself.choi_to_chi(choi, B_choi, n)
        chi_stddv = np.zeros_like(chi)
        choi_stddv = np.zeros_like(chi)
        print('Warning: no standard deviation calculated!')
    else:
        print('Wrong method supplied: %s is not a valid option.' %(method))
        return None
    return ((chi,chi_stddv),(choi,choi_stddv))

def _fit_tomodata_qiskit_(tomo_data, method=None):
    '''
    Use the qiskit functions to fit the tomography data. There are two methods:
        'magic' (standard): This fits the tomography data to a choi matrix that is completely positive and has trace 1.
        For more info of the fitting method, see the documentation of qiskit.tools.qcvv.tomography.fit_tomography_data
        There might be problems with the trace preservation-qualities of the matrix when using this method.
        
        'leastsq' : This fits the data to a Choi matrix via simple linear inversion.
        There is no guarantee on CP (so in almost all cases the Choi matrix will not be CP)
        Therefore different methods have to be used to make the process CP
    Returned is a choi matrix corresponding to the data.
    '''
    if method == None:
        choi_fit = tomo.fit_tomography_data(tomo_data, options={'trace': 1})
    else:
        choi_fit = tomo.fit_tomography_data(
            tomo_data, method, options={'trace': 1})
    return choi_fit


def _fit_tomodata_own_(tomo_data, tomo_set, n, stddv = True):
    '''
    Use own functions to fit the tomography data to a chi matrix.
    For an input pauli P_i and an measurement basis P_j, the function first
        - First rewrites the results in tomo_data into a vector lambda.
            The vector lambda is of the form l(ij) = tr[P_jL(P_i)],
            where L(|p><p|) is the output state of the system on input state |p><|p|.
            As such, L(P_i)] is the (weighted) sum of the positive and negative eigenspace states of P_i:
            See get_lambda_from_meas() for more details
        - Then the matrix A is calculated:
            A(ij,mn) = tr(P_j P_m P_i P_n). See get_A_mat() for more details.
            A is full rank so invertible.
        - The matrix A links chi with lambda:
            l(ij) = sum(m,n) chi(m,n) tr(P_j P_m P_i P_n);
            a vetorized lambda and chi give: l = A*chi.
            Via linear inversion using the numpy.linalg.solve method chi is obtained.
        The indices of A are like:
        row    ij = j + (i*j_total)
        column mn = n + (m*n_total)
    Furthermore, from the standard deviation as provided by get_lambda_from_meas(),
    the standard deviation on chi is calculated. For a single element of chi:
        stddv_chi(mn) = (sum(ij) ((Ainv)^2)(l_stddv)^2)^(1\2)
        This is calculated by first taking the elementwise square of Ainv and l_stddv,
        and then taking the matrix product of Ainv^2 and l_sttdv^2
    The function then returns a tuple (chi, chi_stddv) both shaped into a square matrix
    '''
    B_chi = tomoself.get_pauli_basis(n)                                         # Get the basis in which chi is expressed
    B_prep = tomoself.get_pauli_basis(n, normalise=False)                       # Get the basis in which the experiments are prepared
    B_meas = tomoself.get_pauli_basis(n, normalise=False)                       # Get the basis in which the measurements are done
    lam, lampau, lamstddv = tomoself.get_lambda_from_meas(tomo_set,             # Get the vectors lambda and lambda_stddv from the tomography data
                                                          tomo_data['data'], n)
    A = tomoself.get_A_mat(B_prep, B_meas, B_chi)                               # Calculate the A matrix from the prep, meas and chi basis
    chivect = np.linalg.solve(A, lam)                                           # Invert to calculate chi in vectorform
    Ainv = np.linalg.inv(A)                                                     # Calculate the inverse of A for error calculations (A is full rank)
    Ainvsq = np.abs(Ainv)*np.abs(Ainv);                                         # Calculate the elementwise square of Ainv
    lamstddvsq = lamstddv*lamstddv;                                             # Calculate the elementwise square of l_stddv
    chistddvvect = np.sqrt(Ainvsq @ lamstddvsq)                                 # Calculate the standard deviation on chi using the method from the description
    chi = np.reshape(chivect, ((2*n)**2, (2*n)**2))                             # Reshape into a square matrix
    chistddv = np.reshape(chistddvvect, ((2*n)**2, (2*n)**2))                   # Reshape into a square matrix
    return chi,chistddv 



#%% Trace preservation functions
def _get_total_prob_(tomo_data):
    '''
    Get the total counts of all individual measurements, and divide them by the total number of shots for every experiment.
    This should give a vector of only ones. If that is the case, then Trace Preservation is guaranteed after linear inversion.
    If this is not the case, then there are measurements unaccounted (e.g. a state was prepared but nothing was measured).
    Then, TP is not guaranteed.
    This function is mainly meant as a back-up test if the check_TP() function returns False.
    It works on the measurement data itself,
    but gives no assertion on the TP-qualities of a chi matrix computed from that data,
    if the method used to calculate the chi matrix is not specified.
    For the method fit_chi_own() the TP should be preserved. For fit_tomodata() this is not always te case, see docstring for further details.
    '''
    meas_data = tomo_data['data']                                               # Get the actual data
    assert type(meas_data) == list
    counts = []
    for meas in meas_data:                                                      # For all experiments (elements in meas_data)
        countsvalues = meas['counts'].values()                                  # Get the counts from the dictionary
        counts.append(sum(list(countsvalues))/meas['shots'])                    # Sum all the counts and divide by #shots. //Should be 1
    return counts

def _get_TPsum_(chi, B_chi):
    '''
    Calculates sum(m,n) chi(m,n) B_n^(dagger)@B_m, which should be I for TP processes.
    '''
    d2 = np.shape(chi)[0]                                                       # d^2, the number of elements in the basis {B}
    iden = np.zeros_like(B_chi[0], dtype='complex')                             # A dxd empty matrix to put all elements of the sum in
    for m in range(d2):
        for n in range(d2):
            iden += chi[m, n]*np.mat(B_chi[n]).H@np.mat(B_chi[m])               # The np.mat class has a .H method that gives the hermitian
    return iden


def check_TP(chi, B_chi, n):
    '''
    Returns True when sum(m,n) chi(m.n) B_n^{dagger} B_m is close to the identity channel using the np.allclose() method, returns False if not.
    '''
    assert np.shape(chi) == ((2**n)**2, (2**n)**2)
    iden = np.eye(2**n, dtype='complex')                                        # Identity matrix to which TPsum should be equal to
    TPsum = _get_TPsum_(chi, B_chi)                                               # The sum of chi(m,n)*B_n^(dagger)*B_m for all n and m
    return np.allclose(TPsum,iden)                                              # Return True iff close

#%% Complete positivity functions
def make_CP(chi, n):
    '''
    This function makes a TP process matrix chi completely positive while preserving trace preservation.
    The function works as follows:
        - The eigenvalues of chi are calculated using the np.linalg.eigvals() method.
        - The dimension of the matrix space d(=2**n) is calculated as the trace of chi
        - If the eigenvalues are all nonnegative, then chi is already CP
        - If not, the function then, for the minimum eigenvalue l_m
            Calculates chi' = chi+abs(l_m)*I. Then chi' has no nonzero eigenvalues:
                chi = PDPinv with D the diagonal matrix of the eigenvalues, P (&Pinv) the transformation matrix to the eigenbasis of chi; PPinv = I
                Then: chi' = PDPinv + abs(l_m)*I = PDPinv + abs(l_m)*IPPinv = PDPinv + P*(abs(l_m)*I)*Pinv = P*(D+abs(l_m)I)*Pinv.
                Thus, chi' now has only nonnegative eigenvalues.
            However, the TP condition for chi reads: sum(m,n) chi(m,n) B_n@B_m = I
            We now have: sum(m,n) chi'(m,n) B_n@B_m = sum(m,n) chi(m,n) B_n@B_m + sum(n) abs(l_m) B_n@B_n= I + d^2*l_m*I = (1+d^2*abs(l_m))*I
            With d^2 the number of elements in the basis {B_n}
            The trace of chi' is d*(1+d^2*abs(l_m)), so to obtain a TP version of chi' the function divides chi' by tr(chi')/d
        The function now returns the CP and TP matrix chi'
    '''
    assert np.shape(chi) == ((2**n)**2, (2**n)**2)                              # Check the dimensions
    mineig = np.min(np.linalg.eigvals(chi))                                     # Calculate the minimum eigenvalue of chi
    trace_chi = np.trace(chi)                                                   # Calculate d
    if mineig < 0:
        chiCP = np.add(chi, (-1)*mineig*np.eye((2*n)**2))                       # Calculate chi'
    return trace_chi*chiCP/np.trace(chiCP)                                      # Multiply by d/tr(chi'), which is division by tr(chi')/d

def check_CP(chi):
    '''
    This function checks whether the matrix chi is a CP-process by checking the eigenvalues for negative numbers.
    This works also for Choi matrices.
    '''
    eig = np.linalg.eigvals(chi)
    if np.around(min(eig),10) >= 0:
        return True
    else: return False
#%% Errors and fidelities
def get_chi_error(chi, chi_bas, U, mode='p'):
    '''
    Calculate the error matrix chi_error associated with the process matrix chi. chi_error is the chi matrix in a different basis {B_m@U}:
        rho_out = sum(m,n) chi_error(m,n) (B_m@U) @ rho_in @ (U^dagger@B_n^dagger)
        Using this relation, chi_error can be calculated from chi straight away:
            chi_error = V@chi@V^(dagger), with V(m,n) = tr(B_m^(dagger)@B_(n)@U^(dagger))/d
        
        The error process can also me modeled to take place before the unitary operation:
        rho_out = sum(m,n) chi_error(m,n) (U@B_m) @ rho_in @ (B_n^dagger@ U^dagger)
        Then:
            chi_error = V@chi@V^(dagger), withV(m,n) is specified by tr(B_m^(dagger)@U^(dagger)@B_(n))
        
        Which order is used is specified by 'mode': 
            'p' (standard)  : first the unitary opertaion, then the error channel
            'm'             : first the error channel, then the unitary operation
        
        For more information on chi_error, see 'Error matrices in quantum process tomography' by A. Korotkov.
    '''
    chi = np.mat(chi)
    U = np.mat(U)
    d = np.trace(chi_bas[0].conj().T*chi_bas[0])
    V = np.mat(np.zeros((len(chi_bas), len(chi_bas))), dtype='complex')
    mc = 0
    for i in range(len(chi_bas)):
        chi_bas[i] = np.mat(chi_bas[i])
    for m in chi_bas:
        nc = 0
        for n in chi_bas:
            if mode == 'p':
                V[mc, nc] = np.trace(m.H @ n @ U.H)/d
            if mode == 'n':
                V[mc, nc] = np.trace(m.H @ U.H @ n)/d
            nc += 1
        mc += 1
    return V @ chi @ V.H

def process_fidelity(chi_error, n):
    '''
    Returns the process fidelity tr(chi@chi_des) for a observed chi and a desired chi_des.
    Since this is equal to tr(chi_I@chi_error) with chi_I the perfect chi for the identity channel,
    the process fidelity is calculated as the topleft element of the error matrix chi_error.
    '''
    return chi_error[0,0]/np.trace(chi_error)                                               # Divide by the total trace of chi to get a number between 0 and 1

def channel_fidelity(chi_error,B_choi, n):
    '''
    Returns the channel fidelity as <Om|Choi_error|
    '''
    choi_error = tomoself.chi_to_choi(chi_error, B_choi, n)
    maxent = tomoself.get_max_ent_2n(n)
    return complex(maxent.H @ choi_error @ maxent)



#%% Useful functions
def trace_dist(dens1,dens2):
    '''
    Calculates the trace distance of two density matrices dens1 and dens2,
    by summing the eigenvalues of the 'difference matrix' and dividing by 2.
    '''
    eigvals = np.linalg.eigvals(dens1-dens2)
    Td = 0
    for val in eigvals:
        Td += np.abs(val)
    return Td /2

#%% SPAM filtering
def filter_chi_meas(chi, B_filter, nq):
    '''
    This function filters SPAM errors out of a chi matrix by linear inversion.
    The total process is modeled as a composition of chi_circuit and chi_measurement. Preparation errors are assumed to be negligible.
    
    Then (a hermitian basis is assumed for brevity): 
        rho_final = sum(m,n) chi_total(m,n) B_m@rho_in@B_n = sum(i,j) chi_meas(i,j) B_i @ (sum(k,l) chi_circuit(k,l) B_k @ rho_in @ B_l) @ B_j
        
        This equaltiy can be described as a matrix product:
            chi_total(mn) = sum(kl) chi_circuit(kl) * B(mn,kl),
            with B(mn,kl) = sum(i,j) chi_meas(i,j) tr(B_m@B_i@B_k) tr(B_n@B_l@B_j)
        Note that the matrix B_filter depends on chi_meas
        The matrix B_filter can be calcualted from chi_meas using the get_measfiltermat() or get_measfiltermatind() functions.
        
    Using the np.linalg.solve() method, chi_circuit can be found as the solution x to chi_total = B@x
    
    After finding chi_circuit as a vector, it is reshaped to be a square matrix and then returned
    '''
    d2 = (2**nq)**2
    chivect = np.reshape(chi, (-1, 1))
    chifilter = np.linalg.solve(B_filter, chivect)
    return np.reshape(chifilter, (d2, d2))

def _get_measfiltermat_(chi_meas, B_chi, nq):
    '''
    Calculate B_filter as used in filter_chi_meas using direct matrix multiplication of all elements.
    This takes an considerable amount of time (~30 min on an old i7-2637M laptop cpu).
    For every element B(mn,kl) = sum(i,j) chi_meas(i,j) tr(B_m@B_i@B_k) tr(B_n@B_l@B_j):
        there is a loop over i and j, calculating the traces tr(B_m@B_i@B_k) and tr(B_n@B_l@B_j),
        by first multiplying the matrices and then taking the trace.
    This is done for every element B(mn,kl) of B
    The indices of B are like:
        row    mn = n + (m*n_total)
        column kl = l + (k*l_total)
    '''
    nctot = len(B_chi)
    lctot = len(B_chi)
    B = np.zeros((len(B_chi)*len(B_chi), len(B_chi)**2), dtype='complex')
    for mc, m in enumerate(B_chi):
        for nc, n in enumerate(B_chi):
            for kc, k in enumerate(B_chi):
                for lc, l in enumerate(B_chi):
                    Bmnkl = 0
                    for ic, i in enumerate(B_chi):
                        for jc, j in enumerate(B_chi):
                            tr1 = np.trace(m@i@k)
                            tr2 = np.trace(n@l@j)
                            Bmnkl += chi_meas[ic, jc] * tr1 * tr2
                    B[nc+(mc*nctot), lc+(kc*lctot)] = Bmnkl
    return B

#%% Error weight ratios
def _get_single_weight_(chi_perror, chi_perror_stddv):
    '''
    Get the sum of all the weights of the single-qubit paulis on the diagonal of chi_perror.
    The diagonal of chi_perror, the pauli twirl, has elements corresponding to 2-qubit and 1-qubit paulis.
    Here, all weights of the 1 qubit paulis are summed absolutely.
    Furthermore, the standard deviation on that sum is calculated as sqrt(sum(i) chi_perror_single(i,i)^2)
    The identity element is intentionally left out of the calculations.
    '''
    single_indices = np.array([0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]) # The indices of the single qubit paulis (leave II out)
    pauli_twirl = np.diag(chi_perror)                                           # Take the Pauli twirl
    pauli_twirl_stddv = np.diag(chi_perror_stddv)                               # Take the Pauli twirl of the standard deviation
    single_weights = np.abs(single_indices * pauli_twirl)                       # Calculate the (absolute) single qubit weights
    single_weights_stddv = np.abs(single_indices * pauli_twirl_stddv)           # Calculate the (absolute) single qubit weight standard deviations
    single_stddv = np.sqrt(single_weights_stddv@single_weights_stddv)           # Calculate the total standard deviation of the sum
    return np.sum(single_weights), single_stddv;                                # Return the sum of the weights and the standard deviation

def _get_multi_weight_(chi_perror, chi_perror_stddv):
    '''
    Get the sum of all the weights of the multi-qubit paulis on the diagonal of chi_perror.
    The diagonal of chi_perror, the pauli twirl, has elements corresponding to 2-qubit and 1-qubit paulis.
    Here, all weights of the 2 qubit paulis are summed absolutely.
    Furthermore, the standard deviation on that sum is calculated as sqrt(sum(i) chi_perror_multi(i,i)^2)
    The identity element is intentionally left out of the calculations.
    '''
    multi_indices  = np.array([0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1]) # The indices of the single qubit paulis (leave II out)
    pauli_twirl = np.diag(chi_perror)                                           # Take the Pauli twirl
    pauli_twirl_stddv = np.diag(chi_perror_stddv)                               # Take the Pauli twirl of the standard deviation
    multi_weights = np.abs(multi_indices * pauli_twirl)                         # Calculate the (absolute) multi qubit weights
    multi_weights_stddv = np.abs(multi_indices * pauli_twirl_stddv)             # Calculate the (absolute) single qubit weight standard deviation
    multi_stddv = np.sqrt(multi_weights_stddv@multi_weights_stddv)              # Calculate the total standard deviation of the sum
    return np.sum(multi_weights), multi_stddv;                                  # Return the sum of the weights and the standard deviation

def get_multi_single_ratio(chi_perror, chi_perror_stddv):
    '''
    Calculate the ratio of the multi qubit errors to the single qubit errors, including the standard deviation.
    The individual weights are calculated using get_single_weight() and get_multi_weight().
    The standard deviation is calculated as dR = abs(R)((dx/X)^2+(dy/Y)^2)^(1/2) for R = X/Y.
    '''
    si_weight,si_weight_stddv = _get_single_weight_(chi_perror, chi_perror_stddv)
    mu_weight,mu_weight_stddv = _get_multi_weight_(chi_perror, chi_perror_stddv)
    ratio = mu_weight/si_weight
    ratio_stddv = np.abs(ratio)*np.sqrt((si_weight_stddv/si_weight)**2+(mu_weight_stddv/mu_weight)**2)
    return ratio, ratio_stddv

####################################################################################
#################################### Not Used ######################################
#def get_measfiltermatind(chi_meas, nq):
#    '''
#    Calculate B_filter using a trace relations of the Pauli group
#    '''
#    indices = [0, 1, 2, 3]
#    d = 2**nq
#    row = np.empty((d**8, 1), dtype='complex')
#    i = 0
#    chi_meas_row = np.reshape(chi_meas, (-1, 1))
#    for Bs in tomoself.itt.product(indices, repeat=nq*4):
#        ij = 0
#        for Bij in tomoself.itt.product(indices, repeat=nq*2):
#            tr1 = tomoself.pf.calc_trace_P2prod([Bs[0:2], Bij[0:2], Bs[4:6]])
#            tr2 = tomoself.pf.calc_trace_P2prod([Bs[2:4], Bij[2:4], Bs[6:8]])
#            row[i] += chi_meas_row[ij]*tr1*tr2
#            ij += 1
#        i += 1
#        print('i is:', i)
#    B = np.reshape(row, ((d)**4, (d)**4))
#    return B
#
#def unit_to_choi(Unitary):
#    '''
#    Calculates the choi matrix from a unitary representaion of a map.
#    Choi  = |U>><<U|, with <<U| the vectorized version of U
#    '''
#    vect = np.reshape(Unitary, (1, -1))
#    return np.mat(vect).H @ np.mat(vect)