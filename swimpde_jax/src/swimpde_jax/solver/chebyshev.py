import numpy as np 
import scipy 

def ls_chebyshev(A, b, s_max, s_min, tol = 1e-8, iter_lim = None, show = False, dtype = np.float64):
    """
    Chebyshev iteration for linear least squares problems. 
    """

    A     = scipy.sparse.linalg.aslinearoperator(A)
    m, n  = A.shape
    
    d     = dtype((s_max*s_max+s_min*s_min)/2.0)
    c     = dtype((s_max*s_max-s_min*s_min)/2.0)
    

    theta   = (1.0-s_min/s_max)/(1.0+s_min/s_max) # convergence rate
    itn_est = int(np.ceil((np.log(tol)-np.log(2))/np.log(theta)))
    if (iter_lim is None) or (iter_lim < itn_est):
        iter_lim = itn_est

    alpha = dtype(0.0)
    beta  = dtype(0.0)

    r     = b.copy()
    x     = np.zeros(n, dtype = dtype)
    v     = np.zeros(n, dtype = dtype)

    if show: 
        print('Number of iterations: ', iter_lim)
    for k in range(iter_lim):

        if k == 0:
            beta  = 0.0
            alpha = 1.0/d
        elif k == 1:
            beta  = -1.0/2.0*(c*c)/(d*d)
            alpha =  1.0*(d-c*c/(2.0*d))
        else:
            beta  = -(c*c)/4.0*(alpha*alpha)
            alpha = 1.0/(d-(c*c)/4.0*alpha)

        v  = A.rmatvec(r) - beta*v
        x += alpha*v
        r -= alpha*A.matvec(v)
    

        if show:
            print('Iteration: ', k+1, 'Residual: ', np.linalg.norm(r))


    return x