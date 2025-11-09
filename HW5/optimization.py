import numpy as np
from numpy.linalg import LinAlgError
import scipy
from scipy.optimize import line_search
from datetime import datetime
from collections import defaultdict


class LineSearchTool(object):
    def __init__(self, method='Wolfe', **kwargs):
        self._method = method
        if self._method == 'Wolfe':
            self.c1 = kwargs.get('c1', 1e-4)
            self.c2 = kwargs.get('c2', 0.9)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Armijo':
            self.c1 = kwargs.get('c1', 1e-4)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Constant':
            self.c = kwargs.get('c', 1.0)
        else:
            raise ValueError('Unknown method {}'.format(method))

    @classmethod
    def from_dict(cls, options):
        if type(options) != dict:
            raise TypeError('LineSearchTool initializer must be of type dict')
        return cls(**options)

    def to_dict(self):
        return self.__dict__

    def line_search(self, oracle, x_k, d_k, previous_alpha=None):
        if self._method == 'Constant':
            return self.c
        
        alpha_0 = previous_alpha if previous_alpha is not None else self.alpha_0
        
        if self._method == 'Armijo':
            return self._armijo_search(oracle, x_k, d_k, alpha_0)
        
        elif self._method == 'Wolfe':
            alpha = line_search(
                f=oracle.func,
                myfprime=oracle.grad,
                xk=x_k,
                pk=d_k,
                gfk=oracle.grad(x_k),
                c1=self.c1,
                c2=self.c2
            )[0]
            
            if alpha is not None:
                return alpha
            else:
                return self._armijo_search(oracle, x_k, d_k, alpha_0)

    def _armijo_search(self, oracle, x_k, d_k, alpha_0):
        phi_0 = oracle.func_directional(x_k, d_k, 0)
        derphi_0 = oracle.grad_directional(x_k, d_k, 0)
        alpha = alpha_0
        
        while oracle.func_directional(x_k, d_k, alpha) > phi_0 + self.c1 * alpha * derphi_0:
            alpha /= 2
            if alpha < 1e-16:
                break
        return alpha


def get_line_search_tool(line_search_options=None):
    if line_search_options:
        if type(line_search_options) is LineSearchTool:
            return line_search_options
        else:
            return LineSearchTool.from_dict(line_search_options)
    else:
        return LineSearchTool()


def gradient_descent(oracle, x_0, tolerance=1e-5, max_iter=10000,
                     line_search_options=None, trace=False, display=False):
    start_time = datetime.now()
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)
    
    grad_0 = oracle.grad(x_0)
    grad_norm_0 = np.linalg.norm(grad_0)
    tolerance_sq = tolerance * grad_norm_0
    
    for k in range(max_iter):
        grad_k = oracle.grad(x_k)
        grad_norm_k = np.linalg.norm(grad_k)
        
        if trace:
            current_time = (datetime.now() - start_time).total_seconds()
            history['time'].append(current_time)
            history['func'].append(oracle.func(x_k))
            history['grad_norm'].append(grad_norm_k)
            if x_k.size <= 2:
                history['x'].append(x_k.copy())
        
        if grad_norm_k <= tolerance_sq:
            return x_k, 'success', history
        
        d_k = -grad_k
        
        alpha_k = line_search_tool.line_search(oracle, x_k, d_k)
        
        if alpha_k is None or np.isinf(alpha_k) or np.isnan(alpha_k):
            return x_k, 'computational_error', history
        
        x_k += alpha_k * d_k
        
        if display and k % 100 == 0:
            print(f"Iteration {k}, f(x) = {oracle.func(x_k):.6f}, grad_norm = {grad_norm_k:.6f}")
    
    grad_final = oracle.grad(x_k)
    if np.linalg.norm(grad_final) <= tolerance_sq:
        return x_k, 'success', history
    else:
        return x_k, 'iterations_exceeded', history


def newton(oracle, x_0, tolerance=1e-5, max_iter=100,
           line_search_options=None, trace=False, display=False):
    start_time = datetime.now()
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)
    
    grad_0 = oracle.grad(x_0)
    grad_norm_0 = np.linalg.norm(grad_0)
    tolerance_sq = tolerance * grad_norm_0
    
    for k in range(max_iter):
        grad_k = oracle.grad(x_k)
        grad_norm_k = np.linalg.norm(grad_k)
        
        if trace:
            current_time = (datetime.now() - start_time).total_seconds()
            history['time'].append(current_time)
            history['func'].append(oracle.func(x_k))
            history['grad_norm'].append(grad_norm_k)
            if x_k.size <= 2:
                history['x'].append(x_k.copy())
        
        if grad_norm_k <= tolerance_sq:
            return x_k, 'success', history
        
        hess_k = oracle.hess(x_k)
        try:
            L, lower = scipy.linalg.cho_factor(hess_k)
            d_k = scipy.linalg.cho_solve((L, lower), -grad_k)
        except (LinAlgError, ValueError) as e:
            return x_k, 'newton_direction_error', history
        
        alpha_k = line_search_tool.line_search(oracle, x_k, d_k, previous_alpha=1.0)
        
        if alpha_k is None or np.isinf(alpha_k) or np.isnan(alpha_k):
            return x_k, 'computational_error', history
        
        x_k += alpha_k * d_k
        
        if display and k % 10 == 0:
            print(f"Iteration {k}, f(x) = {oracle.func(x_k):.6f}, grad_norm = {grad_norm_k:.6f}")
    
    grad_final = oracle.grad(x_k)
    if np.linalg.norm(grad_final) <= tolerance_sq:
        return x_k, 'success', history
    else:
        return x_k, 'iterations_exceeded', history
    
    
    