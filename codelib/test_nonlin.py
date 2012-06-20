# coding: utf-8
#
#   tests for codelib.nonlin
#
#   Copyright (C) 2012 Denis Laxalde <denis@laxalde.org>
#
#   Licensed under the GNU GPL version 3 or higher.
#
import unittest
import numpy as np
from scipy import sparse
import nonlin

class TestProblems(unittest.TestCase):
    def test_cubicsystem(self, n=20):
        """System of n equation defined by A * x**3 - b = 0"""
        f = lambda x, A, B: np.dot(A, x**3) - B
        df = lambda x, A, B: np.diag(3*np.dot(A, x**2))
        a = np.identity(n)
        b = np.random.rand(n)*10
        xi = np.ones(n)
        res = nonlin.root(f, xi, (a, b), jac=df,
                          options=dict(rtol=1e-10, maxit=20,
                                       disp={'err':0,'it':0}, nonlin=2,
                                       ordi=1))
        self.assertTrue(res['success'], res['message'])
        self.assertTrue(np.linalg.norm(res.fvec) < 1e-6)

    def test_semicon(self):
        """ root: 2D semiconductor device simulation """
        def fun(x, alpha, D, ni, V):
            f = np.zeros(6)
            f[0] = (np.exp(alpha * (x[2] - x[0])) -
                    np.exp(alpha * (x[0] - x[1])) - D / ni)
            f[1] = x[1]
            f[2] = x[2]
            f[3] = (np.exp(alpha * (x[5] - x[3])) -
                    np.exp(alpha * (x[3] - x[4])) + D / ni)
            f[4] = x[4] - V
            f[5] = x[5] - V
            return f

        def jac(x, alpha, D, ni, V):
            j = np.zeros((6, 6))
            j[0, 0] = (-alpha * np.exp(alpha * (x[2] - x[0])) -
                       alpha * np.exp(alpha * (x[0] - x[1])))
            j[0, 1] = alpha * np.exp(alpha * (x[0] - x[1]))
            j[0, 2] = alpha * np.exp(alpha * (x[2] - x[0]))
            j[1, 1] = 1.
            j[2, 2] = 1.
            j[3, 3] = (-alpha * np.exp(alpha * (x[5] - x[3])) -
                       alpha * np.exp(alpha * (x[3] - x[4])))
            j[3, 4] = alpha * np.exp(alpha * (x[3] - x[4]))
            j[3, 5] = alpha * np.exp(alpha * (x[5] - x[3]))
            j[4, 4] = 1.
            j[5, 5] = 1.
            return j

        alpha, ni, V, D = 38.683, 1.22e10, 100., 1e17
        args = (alpha, D, ni, V)
        xi = np.ones(6)
        res = nonlin.root(fun, xi, args, jac,
                          options=dict(rtol=1e-10, disp={'err': 0,'it': 0},
                                       nonlin=4))
        self.assertTrue(res['success'], res['message'])

    #@unittest.expectedFailure
    # this test fails if extra arguments are passed.
    def test_chemeq(self):
        """ NLEQ1S: Chemical equilibrium problem. """
        def fun(x, a=1):
            f = [x[0] + x[1] + x[3] - 1.0e-3,
                 x[4] + x[5] - 55.0,
                 x[0] + x[1] + x[2] + 2.0 * x[4] + x[5] - 110.001,
                 x[0] - 1.0e-1 * x[1],
                 x[0] - 1.0e4 * x[2] * x[3],
                 1.0e-14 * x[4] - 55.0 * x[2] * x[5]]

            return np.array(f)

        def jac(x, a=1):
            df   = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                    1.0, 1.0, 2.0, 1.0, 1.0, -0.1,
                    1.0, -1.0e4 * x[3], -1.0e4 * x[2],
                    -55.0 * x[5], 1.0e-14, -55.0 * x[3]]
            irow = [1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 4, 4, 5, 5, 5, 6, 6, 6]
            icol = [1, 2, 4, 5, 6, 1, 2, 3, 5, 6, 1, 2, 1, 3, 4, 3, 5, 6]
            # zero indexing
            i = [k-1 for k in irow]
            j = [k-1 for k in icol]
            return sparse.coo_matrix((df, [i, j]))

        xi = [3.0e-4, 3.0e-4, 27.5, 3.0e-4, 27.5, 27.5]
        res = nonlin.root(fun, xi, jac=jac,
                          options=dict(rtol=1e-5, disp={'err': 0,'it': 0},
                                       sparse=True, nnz=18, nonlin=3))
        self.assertTrue(res['success'], res['message'])


if __name__ == '__main__':
    unittest.main()
