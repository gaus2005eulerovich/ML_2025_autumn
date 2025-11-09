import numpy as np
import scipy
from scipy.special import expit
import scipy.sparse


class BaseSmoothOracle(object):
    def func(self, x):
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        raise NotImplementedError('Grad oracle is not implemented.')
    
    def hess(self, x):
        raise NotImplementedError('Hessian oracle is not implemented.')
    
    def func_directional(self, x, d, alpha):
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        return np.squeeze(self.grad(x + alpha * d).dot(d))


class QuadraticOracle(BaseSmoothOracle):
    def __init__(self, A, b):
        if not scipy.sparse.isspmatrix_dia(A) and not np.allclose(A, A.T):
            raise ValueError('A should be a symmetric matrix.')
        self.A = A
        self.b = b

    def func(self, x):
        return 0.5 * np.dot(self.A.dot(x), x) - self.b.dot(x)

    def grad(self, x):
        return self.A.dot(x) - self.b

    def hess(self, x):
        return self.A 


class LogRegL2Oracle(BaseSmoothOracle):
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA
        self.b = b
        self.regcoef = regcoef
        self.m = len(b)

    def func(self, x):
        Ax = self.matvec_Ax(x)
        b_Ax = self.b * Ax
        log_exp_term = np.logaddexp(0, -b_Ax)
        data_loss = np.mean(log_exp_term)
        reg_loss = 0.5 * self.regcoef * np.dot(x, x)
        return data_loss + reg_loss

    def grad(self, x):
        Ax = self.matvec_Ax(x)
        b_Ax = self.b * Ax
        p = expit(b_Ax)
        grad_data = self.matvec_ATx(-self.b * (1 - p)) / self.m
        grad_reg = self.regcoef * x
        return grad_data + grad_reg

    def hess(self, x):
        Ax = self.matvec_Ax(x)
        b_Ax = self.b * Ax
        p = expit(b_Ax)
        d = p * (1 - p)
        hess_data = self.matmat_ATsA(d) / self.m
        hess_reg = self.regcoef * np.eye(len(x))
        return hess_data + hess_reg


class LogRegL2OptimizedOracle(LogRegL2Oracle):
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        super().__init__(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)
        self._x_cache = None
        self._Ax_cache = None
        self._d_cache = None
        self._Ad_cache = None
        self._xdx_cache = None
        self._Axdx_cache = None

    def _update_cache(self, x, d=None):
        if self._x_cache is None or not np.array_equal(x, self._x_cache):
            self._x_cache = x.copy()
            self._Ax_cache = self.matvec_Ax(x)
        
        if d is not None and (self._d_cache is None or not np.array_equal(d, self._d_cache)):
            self._d_cache = d.copy()
            self._Ad_cache = self.matvec_Ax(d)

    def func(self, x):
        self._update_cache(x)
        b_Ax = self.b * self._Ax_cache
        log_exp_term = np.logaddexp(0, -b_Ax)
        data_loss = np.mean(log_exp_term)
        reg_loss = 0.5 * self.regcoef * np.dot(x, x)
        return data_loss + reg_loss

    def grad(self, x):
        self._update_cache(x)
        b_Ax = self.b * self._Ax_cache
        p = expit(b_Ax)
        grad_data = self.matvec_ATx(-self.b * (1 - p)) / self.m
        grad_reg = self.regcoef * x
        return grad_data + grad_reg

    def hess(self, x):
        self._update_cache(x)
        b_Ax = self.b * self._Ax_cache
        p = expit(b_Ax)
        d = p * (1 - p)
        hess_data = self.matmat_ATsA(d) / self.m
        hess_reg = self.regcoef * np.eye(len(x))
        return hess_data + hess_reg

    def func_directional(self, x, d, alpha):
        self._update_cache(x, d)
        x_new = x + alpha * d
        Ax_new = self._Ax_cache + alpha * self._Ad_cache
        b_Ax_new = self.b * Ax_new
        log_exp_term = np.logaddexp(0, -b_Ax_new)
        data_loss = np.mean(log_exp_term)
        reg_loss = 0.5 * self.regcoef * np.dot(x_new, x_new)
        self._xdx_cache = x_new
        self._Axdx_cache = Ax_new
        return data_loss + reg_loss

    def grad_directional(self, x, d, alpha):
        self._update_cache(x, d)
        x_new = x + alpha * d
        
        if (self._xdx_cache is not None and 
            np.array_equal(x_new, self._xdx_cache) and 
            self._Axdx_cache is not None):
            Ax_new = self._Axdx_cache
        else:
            Ax_new = self._Ax_cache + alpha * self._Ad_cache
            self._xdx_cache = x_new
            self._Axdx_cache = Ax_new
        
        b_Ax_new = self.b * Ax_new
        p = expit(b_Ax_new)
        
  
        part1 = np.dot(self._Ad_cache, -self.b * (1 - p)) / self.m
        part2 = self.regcoef * np.dot(d, x_new)
        return part1 + part2

def create_log_reg_oracle(A, b, regcoef, oracle_type='usual'):
    if scipy.sparse.issparse(A):
        def matvec_Ax(x):
            return A.dot(x)
        
        def matvec_ATx(x):
            return A.T.dot(x)
        
        def matmat_ATsA(s):
            return A.T.dot(scipy.sparse.diags(s).dot(A))
    else:
        def matvec_Ax(x):
            return A @ x
        
        def matvec_ATx(x):
            return A.T @ x
        
        def matmat_ATsA(s):
            return A.T @ (s[:, np.newaxis] * A)

    if oracle_type == 'usual':
        oracle_class = LogRegL2Oracle
    elif oracle_type == 'optimized':
        oracle_class = LogRegL2OptimizedOracle
    else:
        raise ValueError('Unknown oracle_type=%s' % oracle_type)
    
    return oracle_class(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)


def grad_finite_diff(func, x, eps=1e-8):
    n = len(x)
    grad = np.zeros(n)
    f_x = func(x)
    
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1
        f_x_eps = func(x + eps * e_i)
        grad[i] = (f_x_eps - f_x) / eps
    
    return grad


def hess_finite_diff(func, x, eps=1e-5):
    n = len(x)
    hess = np.zeros((n, n))
    f_x = func(x)
    
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1
        f_x_i = func(x + eps * e_i)
        
        for j in range(n):
            e_j = np.zeros(n)
            e_j[j] = 1
            f_x_j = func(x + eps * e_j)
            f_x_ij = func(x + eps * e_i + eps * e_j)
            hess[i, j] = (f_x_ij - f_x_i - f_x_j + f_x) / (eps * eps)
    
    return hess

