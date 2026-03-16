import numpy as np 
import scipy 
import time 
from swimpde_jax.solver.chebyshev import ls_chebyshev


def lsrn(A, b, gamma=2.0, tol=np.finfo(float).eps, rcond=-1, block_size = -1, show = False, solver = 'lsqr', dtype=np.float64):
    """ Implementation of LSRN algorithm for solving linear least squares problem 
    
    Arguments:
    A : {matrix, sparse matrix, ndarray, LinearOperator} of size m-by-n
    b : (m,k) ndarray, k right-hand sides
    gamma : float (>1), oversampling factor. Default is 2.0
    tol : float, tolerance such that norm(A*x-A*x_opt)<tol*norm(A*x_opt)
    rcond : float, reciprocal condition number, 

    Returns:
    x : (n,) ndarray, the min-length solution
    r : int, the rank of A
    timing : dict, 
    """
    A = scipy.sparse.linalg.aslinearoperator(A)
    m, n = A.shape
    rng = np.random.default_rng() # random number generator
    
    if rcond < 0:
        rcond = np.min([m,n])*np.finfo(float).eps

    timing = {'mult': 0.0, 'svd': 0.0, 'solver': 0.0 }
    if m >= n:                           # over-determined
        s = int(np.ceil(gamma*n))
        
        if block_size < 0:
            block_size = s
        
        # multiply A with G, but G is too large to fit in memory
        t_start = time.time()        

        # this is row-splitting
        number_of_blocks = int(np.ceil(s/block_size))
        print('Number of blocks for the generation of G:', number_of_blocks)
        As = np.empty((s, n), dtype=dtype)

        for i in range(number_of_blocks):
            print(f"Block {i+1}")
            block_begin = i*block_size
            block_end = np.min([(i+1)*block_size, s])
            block_length = block_end-block_begin

            # split G row-wise
            G = rng.standard_normal(size=(block_length, m), dtype = dtype)
            As[block_begin:block_end,:] = A.rmatmat(G.T).T

        timing['mult'] = time.time()-t_start

        t_start = time.time()
        print('SVD solve')
        S, V = scipy.linalg.svd(As, full_matrices=False)[1:3] # only need the singular values and right singular vectors
        # determine the rank
        r_tol = S[0]*rcond
        r = np.sum(S>r_tol)
        timing['svd'] = time.time()-t_start

        # find the preconditioner
        print('Preconitioning')
        N = V[:r,:].T / S[:r]
        gamma    = 1.0*s/r                             # re-estimate gamma
        condest  = (np.sqrt(gamma)+1.0)/(np.sqrt(gamma)-1.0) # condition number of AN
        iter_lim = np.ceil(-2*(np.log(tol)-np.log(2))/np.log(gamma))
        time_start = time.time()
        # define AN as a linear operator
        preconditioned_A = scipy.sparse.linalg.LinearOperator((m, N.shape[1]), matvec=lambda x: A.matvec(N@x), rmatvec=lambda x: N.T @ (A.rmatvec(x)), dtype=dtype)


        if len(b.shape) == 1:
            k = 1 
            b = b.reshape(-1,1)
        else:
            k = b.shape[1]
        
        x = np.empty((N.shape[0], k), dtype=dtype)
        for i in range(b.shape[1]): 
            if solver == 'lsqr':
                result = scipy.sparse.linalg.lsqr(preconditioned_A, b[:,i], iter_lim = iter_lim, atol = tol/condest, btol = tol/condest, show=show)
                x[:,i] = N@result[0]
            elif solver == 'chebyshev':
                p_fail = 0.1  # failure rate
                t = np.sqrt(-2.0*np.log(p_fail/2.0))
                s_max = 1.0/(np.sqrt(s)-np.sqrt(r)-t)
                s_min = 1.0/(np.sqrt(s)+np.sqrt(r)+t)
                result = ls_chebyshev(preconditioned_A, b[:,i], s_max, s_min, tol = tol, iter_lim = iter_lim, show = show)
                x[:,i] = N @ result
            
            #print('Number of iterations for dimension ', i, result[2])

        timing['solver'] = time.time()-time_start

    else: 
        s = int(np.ceil(gamma*m))
        
        if block_size < 0:
            block_size = s
        
        t_start = time.time()
        number_of_blocks = int(np.ceil(s/block_size))
        As = np.empty((m, s), dtype=dtype)
        for i in range(number_of_blocks):
            block_begin = i*block_size
            block_end = np.min([(i+1)*block_size, s])
            block_length = block_end-block_begin
            G = rng.standard_normal(size=(n, block_length), dtype = dtype)
            As[:,block_begin:block_end] = A.matmat(G)
        
        timing['mult'] = time.time()-t_start

        t_start = time.time()
        U, S = scipy.linalg.svd(As, full_matrices=False)[0:2]
        # determine the rank
        r_tol = S[0]*rcond
        r = np.sum(S>r_tol)
        timing['svd'] = time.time()-t_start

        # find the preconditioner
        M = U[:,:r] / S[:r]

        gamma    = 1.0*s/r
        condest  = (np.sqrt(gamma)+1.0)/(np.sqrt(gamma)-1.0)
        iter_lim = np.ceil(-2*(np.log(tol)-np.log(2))/np.log(gamma))
        time_start = time.time()
        # define M.T @ A as a linear operator. right hand side b will be multiplied with M.T
        preconditioned_A = scipy.sparse.linalg.LinearOperator((r, n), matvec=lambda x: M.T @ (A.matvec(x)), rmatvec=lambda x: A.rmatvec(M @ x), dtype=dtype)

        # take care if the user gives only one right-hand side
        if len(b.shape) == 1:
            k = 1 
            b = b.reshape(-1,1)
        else:
            k = b.shape[1]
        
        x = np.empty((n, k), dtype=dtype)

        for i in range(b.shape[1]):
            if solver == 'lsqr':
                result = scipy.sparse.linalg.lsqr(preconditioned_A, M.T @ b[:,i], iter_lim = iter_lim, atol = tol/condest, btol = tol/condest, show=show)
                x[:,i] = result[0]
            elif solver == 'chebyshev':
                p_fail = 0.1
                t = np.sqrt(-2.0*np.log(p_fail/2.0))
                s_max = 1.0/(np.sqrt(s)-np.sqrt(r)-t)
                s_min = 1.0/(np.sqrt(s)+np.sqrt(r)+t)
                result = ls_chebyshev(preconditioned_A, M.T @ b[:,i], s_max, s_min, tol = tol, iter_lim = iter_lim, show = show, dtype = dtype)
                x[:,i] = result
        
        timing['solver'] = time.time()-time_start
                
    return x, r, timing
