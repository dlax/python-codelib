# coding: utf-8
#
#   Interface to algorithms for Nonlinear systems of equations and least
#   squares problems
#
#   Copyright (C) 2012 Denis Laxalde <denis@laxalde.org>
#
#   Licensed under the GNU GPL version 3 or higher.
#
"""
Nonlinear systems of equations
==============================

Functions
---------
- root: solve a system of nonlinear equations
"""
import numpy as np
from scipy import sparse
from scipy.optimize import Result
import _nleq1
import _nleq1s
from warnings import warn

__all__ = ['root']

def _print_options(name, value):
    """Returns a formatted message corresponding to option 'name'"""
    return ':'.join((name.rjust(20),
                     ' {0}'.format(value)))

def _set_options(method, n, nfmax=None, jacobian=False, **opts):
    """Returns general options 'gopt' and some solver specific options
    depending on the method"""
    genopt = ('rtol', 'maxit')
    # default options, updated using kw params
    gopt = dict(rtol=1e-10, maxit=50)
    gopt.update(((k, v) for k, v in opts.iteritems() if k in genopt))
    if method == '1': # NLEQ1 solver
        # iopt
        iopt = np.zeros(50, dtype=int) # default options
        if jacobian:
            iopt[2] = 1
        else:
            iopt[2] = 2
        iopt[1] = 1 * opts.get('stepwisemode', False)
        if 'jacgen' in opts:
            pass
        xscal = opts.get('xscal')
        if xscal is not None:
            iopt[8] = 1
        else:
            iopt[8] = 0
        disp = opts.get('disp', {})
        iopt[10] = disp.get('err', 0)
        iopt[12] = disp.get('it', 0)
        iopt[30] = opts.get('nonlin', 3)
        iopt[31] = opts.get('broydupdate', 0)
        # XXX: review IOPT options after 31
        if 'ordi' in opts:
            iopt[32] = opts['ordi']

        # iwk
        iwk = np.zeros(n+50, dtype=int)
        iwk[30] = gopt.get('maxit', 0)
        # rwk
        if iopt[31]:
            nbroy = n
        else:
            nbroy = 0
        rwk = np.zeros((n + nbroy + 13)*n + 61)
        if 'fcmin' in opts:
            rwk[21] = opts['fcmin']

        return gopt, iopt, iwk, rwk
    elif method == '1s':
        # iopt
        iopt = np.zeros(50, dtype=int) # default options
        iopt[1] = 1 * opts.get('stepwisemode', False)
        xscal = opts.get('xscal')
        if xscal is not None:
            iopt[8] = 1
        else:
            iopt[8] = 0
        disp = opts.get('disp', {})
        iopt[10] = disp.get('err', 0)
        iopt[12] = disp.get('it', 0)
        iopt[14] = disp.get('sol', 0)
        iopt[16] = disp.get('lin', 0)
        iopt[18] = disp.get('time', 0)
        iopt[30] = opts.get('nonlin', 3)
        iopt[31] = opts.get('broydupdate', 0)
        # XXX: review IOPT options after 31
        iopt[36] = opts.get('fixpt', 0)

        # iwk
        iwk = np.zeros(11 * n + 62 + 12 * nfmax, dtype='int')
        iwk[30] = gopt.get('maxit', 50)
        # rwk
        if iopt[31]:
            nbroy = max(int(nfmax/n)+1, 10)
        else:
            nbroy = 0
        rwk = np.zeros(6 * nfmax + (12 + nbroy) * n + 68, dtype='double')
        if 'fcmin' in opts:
            rwk[21] = opts['fcmin']

        return gopt, iopt, iwk, rwk
    else:
        raise ValueError('The chosen method (%s) does not exist' % method)

def root(fun, x0, args=(), jac=None, options=None):
    """Solve a nonlinear system of equations defined by fun(x, *args) = 0

    Parameters
    ----------
    fun : callable
        Objective function to find a root of.
    x0 : ndarray
        Initial guess.
    args : tuple
        Optional arguments for fun and jac.
    jac : callable
        Jacobian of the objective function. Must return a sparse matrix if
        options['sparse']==True, an array otherwise.
    options : dict
        Solver options.
            'rtol'  : relative precision
            'maxit' : maximum number of iterations allowed
            'sparse' : bool
                If False, NLEQ1 solver is used:
                <http://www.zib.de/en/numerik/software/ant/nleq1.html>
                If True NLEQ1S solver is used:
                <http://www.zib.de/en/numerik/software/ant/nleq1s.html>
            'nnz' : int
                Maximum number of nonzero elements of the Jacobian matrix for
                method '1s' only. If None, this will be evaluated from calling
                `jac`.
            'xcal'  : scaling vector for vector x
            'broydupdate' : if non-zero Broyden approximation are allowed.
            'stepwisemode' : if non-zero, stepwise mode is used.
            'disp' : dict
                Control the verbosity at different level of the algorithm. From 0
                to 3 usually. (Some options are not avaible in the dense
                version.)
                    'err': errors
                    'it': iterations
                    'sol': solutions
                    'lin': linear solver
                    'time': time
            'nonlin' : level of nonlinearity of the problem
                    (1: linear, 2: midly nonlinear, 3: highly nonlinear,
                     4: extremely nonlinear)
            'fixpt' : Fixed sparse pattern option: if 0 the sparse pattern
                of the Jacobian may vary for different iterates, if 1 he
                sparse pattern of the Jacobian is fixed for all calls.

    Returns
    -------
    res : Result
        Solution or result from the last iteration if the solver did not
        converge.

    """
    if options is None:
        options = dict()
    sparse_pb = options.get('sparse', False)
    x = np.asarray(x0)
    n = len(x0)
    if x.ndim != 1:
        raise ValueError('Only rank-1 variables are allowed')
    if not sparse_pb:
        gopt, iopt, iwk, rwk = _set_options('1', n,
                                            jacobian=(jac is not None),
                                            **options)
        if iopt[8] == 1:
            xscal = gopt['xscal']
        else:
            xscal = np.zeros(n)
        rtol = gopt['rtol']
        (xs, xscal, rtol, iopt,
         ierr, iwk, rwk) = _nleq1.nleq1(fun, jac, x, xscal, rtol, iopt,
                                        iwk, rwk, fcn_extra_args=args,
                                        jac_extra_args=args)

    else:
        if jac is None:
            ValueError('The jacobian function is needed for 1s method.')

        # wrap the Jacobian function to retrieve the sparse matrix info
        def jac1(x, *args):
            J = jac(x, *args).tocoo()
            return J.data, J.row+1, J.col+1, J.nnz

        try:
            nnz = options['nnz']
        except KeyError:
            nnz = jac1(x, *args)[3]

        gopt, iopt, iwk, rwk = _set_options('1s', n, nnz,
                                            jacobian=(jac is not None),
                                            **options)
        if iopt[8] == 1:
            xscal = gopt['xscal']
        else:
            xscal = np.zeros(n)
        rtol = gopt['rtol']

        (xs, xscal, rtol, iopt,
         ierr, iwk, rwk) = _nleq1s.nleq1s(nnz, fun, jac1, x, xscal, rtol,
                                          iopt, iwk, rwk,
                                          fcn_extra_args=args,
                                          jac_extra_args=args)

    res = Result(x=xs, fvec=fun(xs, *args), nit=iwk[0], nfev=iwk[3],
                 njev=iwk[4], conv=rwk[16], liwk=iwk[17], lrwk=iwk[18],
                 ifail=iwk[22], status=ierr, success=ierr==0)
    if ierr > 5:
        warn('Internal error in NLEQ1S solver ' '(error code: %d)' %
             ierr, RuntimeWarning)
    try:
        res['message'] = {
            0: 'A solution was found at the specified tolerance.',
            1: 'Termination: Jacobian matrix became singular',
            2: 'Termination: maximum number of iterations exceeded.',
            3: 'Termination: damping factor has become too small',
            4: 'Warning: Superlinear or quadratic convergence '
               'slowed down near the solution.',
            5: 'Warning: Iteration stopped with termination '
               'criterion satisfied, but no superlinear or '
               'quadratic convergence has been indicated yet.'
            }[ierr]
    except KeyError:
        res['message'] = ('Internal error in NLEQ1S solver '
                          '(error code: %d)' % ierr)
    return res
