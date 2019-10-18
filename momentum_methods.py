# %matplotlib inline
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import datetime
import math
from utils import *
# plt.style.use('seaborn-white')


gamma = 500

def griewank(x):
    n = len(x)
    indices = np.array(range(1, n+1))
    sum_term = np.sum(x*x/gamma)
    prod_term = np.prod(np.cos(x/np.sqrt(indices)))
    return 1 + sum_term - prod_term


def plot_griewank():
    n = 100
    x = np.linspace(x_min, x_max, n)
    y = np.linspace(x_min, x_max, n)

    X, Y = np.meshgrid(x, y)
    Z = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            x = np.array([X[i][j], Y[i][j]])
            Z[i][j] = griewank(x, gamma)

    plt.contour(X, Y, Z, cmap='RdGy')
    plt.colorbar()


def grad_griewank(x):
    n = len(x)
    indices = np.array(range(1, n+1))
    grad = [2*x[i]/gamma + np.prod(np.cos(x/np.sqrt(indices))) * (1/np.cos(x[i] / np.sqrt(i+1))) * np.sin(x[i] / np.sqrt(i+1)) * (1.0 / np.sqrt(i+1)) for i in range(n)]
    return np.array(grad)


def hessian_griewank(x):
    hessian = np.zeros([2, 2])
    hessian[0][0] = 2.0/gamma + np.cos(x[1]/np.sqrt(2))*np.cos(x[0])
    hessian[0][1] = hessian[1][0] = (-1/np.sqrt(2))*np.sin(x[0])*np.sin(x[1]/np.sqrt(2))
    hessian[1][1] = (2.0/gamma) + 0.5*np.cos(x[0])*np.cos(x[1]/np.sqrt(2))
    return hessian


def grad_is_approx_zero(grad, epsilon=1e-5):
    for grad_val in grad:
        if abs(grad_val) > epsilon:
            return False
    return True

def ebls(x, d, alpha_start=1, c1=0.1, c2=0.3, epsilon=1e-5, fun=griewank, grad=grad_griewank):
    L = 0.0
    U = np.inf # infinity
    alpha = alpha_start
    x_curr = x
    n_iter = 1
    grad_curr = -1 * d
    f_curr = fun(x_curr)
    while n_iter <= 25:
        n_iter += 1
        f_next = fun(x_curr + alpha*d)
        if f_next > (f_curr + c1*alpha*np.matmul(grad_curr.T, d)):
            U = alpha
            alpha = (U + L)/2.0
        else:
            grad_next = grad(x_curr + alpha*d)
            if np.matmul(grad_next.T, d) < c2*(np.matmul(grad_curr.T, d)):
                L = alpha
                if U >= 1e100:
                    alpha = 2*L
                else:
                    alpha = (L + U)/2.0
            else:
                break

    x_next = x_curr + alpha*d
    return [alpha, x_next, f_next, grad_next]


def steepest_descent(x0, alpha=1, epsilon=1e-5, n_iter=10000, debug=False, gamma=500, use_ebls=False, fun=griewank, grad=grad_griewank):
    x_vals = []
    obj_vals = []
    grad_vals = []

    n = len(x0)
    x_curr = x0
    grad_curr = grad(x_curr)
    
    for i in range(n_iter):  
        d = -1.0 * grad_curr

        x_vals.append(x_curr)
        obj_vals.append(fun(x_curr))
        grad_vals.append(grad_curr)

        if grad_is_approx_zero(grad_curr, epsilon):
            if debug:
                print("Solution found!", x_curr, fun(x_curr), grad_curr)
            break
        if use_ebls:
            alpha, x_next, f_next, grad_next = ebls(x_curr, d)
        else:
        	x_next = x_curr - alpha*grad_curr
        	grad_next = grad(x_next)

        x_curr = x_next
        grad_curr = grad_next
    results_df = pd.DataFrame({'x': x_vals, 'obj': obj_vals, 'grad': grad_vals})
    return x_curr, fun(x_curr), results_df


def get_rho_k(rho_k_minus_1=0):
    a = -1.0
    b = (rho_k_minus_1**2 - 1)
    c = 1.0
    rho_k = (-b - math.sqrt(b**2 - 4*a*c))/2*a
    return rho_k


def nesterov(x0, alpha=1, beta=0.98, epsilon=1e-5, n_iter=1000, debug=False, gamma=500, use_ebls=False, rho=0, fun=griewank, grad=grad_griewank):
    x_vals = []
    obj_vals = []
    grad_vals = []

    n = len(x0)
    x_curr = x0
    x_prev = x0
    rho_prev = rho
    
    for i in range(n_iter):

        rho_curr = get_rho_k(rho_prev)
        beta = rho_curr*(rho_prev**2)
        rho_prev = rho_curr

        y_curr = x_curr + beta*(x_curr - x_prev)
        grad_curr = grad(y_curr)

        x_vals.append(x_curr)
        obj_vals.append(fun(x_curr))
        grad_vals.append(grad)

        if grad_is_approx_zero(grad_curr, epsilon):
            if debug:
                print("Solution found!", x_curr, fun(x_curr))
            break
        x_next = y_curr - alpha*grad_curr
        x_prev = x_curr
        x_curr = x_next
    results_df = pd.DataFrame({'x': x_vals, 'obj': obj_vals, 'grad': grad_vals})
    return x_curr, fun(x_curr), results_df


def heavy_ball(x0, alpha=1, beta=0.98, epsilon=1e-5, n_iter=10000, debug=False, gamma=500, use_ebls=False, fun=griewank, grad=grad_griewank):
    x_vals = []
    obj_vals = []
    grad_vals = []

    n = len(x0)
    x_curr = x0
    x_prev = x0
    
    for i in range(n_iter):
        beta=0.98
        grad_curr = grad(x_curr)

        x_vals.append(x_curr)
        obj_vals.append(fun(x_curr))
        grad_vals.append(grad_curr)

        if grad_is_approx_zero(grad_curr, epsilon):
            if debug:
                print("Solution found!", x_curr, fun(x_curr))
            break
        
        if use_ebls:
            alpha, x_next, f_next, grad_next = ebls(x_curr, -1*grad_curr)
        
        x_next = x_curr - alpha*grad_curr + beta*(x_curr - x_prev)
        
        if debug:
            print("Momentum!", x_curr-x_prev)
        
        x_prev = x_curr
        x_curr = x_next
    results_df = pd.DataFrame({'x': x_vals, 'obj': obj_vals, 'grad': grad_vals})
    return x_curr, fun(x_curr), results_df


def beta_grid_search(n_iter=100, beta_values=None, gamma=500, alpha=1):
    beta_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.87, 0.89, 0.90, 0.91,
            0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.999]
    results_dict = {'beta': [], 'dist_from_start': [], 'dist_to_origin': [], 'f_opt': [], 'start_distance': []}
    for beta in beta_values:
        print "beta:", beta
        df = run_griewank_test('heavy_ball', gamma=gamma, n_iter=n_iter, alpha=alpha, beta=beta, use_ebls=False)
        results_dict['beta'].append(beta)
        results_dict['dist_from_start'].append(df.dist_from_start.mean())
        results_dict['dist_to_origin'].append(df.dist_to_origin.mean())
        results_dict['f_opt'].append(df.f_opt.mean())
        results_dict['start_distance'].append(df.start_distance.mean())
    return pd.DataFrame(results_dict)


def compare_descent_algorithms(gamma_spec=500, x_min=-20, x_max=20, n_iter=100, max_iter=10000, alpha=1, beta=0.98):
    global gamma
    gamma = gamma_spec

    def save_results(results_list, x_0, x_opt, f_opt, iter_to_convergence):
        results_list['dist_from_start'].append(np.linalg.norm(x_opt - x_0))
        results_list['dist_to_origin'].append(np.linalg.norm(x_opt))
        results_list['f_opt'].append(f_opt)
        results_list['iter_to_convergence'].append(iter_to_convergence)
        results_list['dist_moved_towards_origin'].append(np.linalg.norm(x_0) - np.linalg.norm(x_opt))

    start_distance = []
    sd_results = {'dist_from_start': [], 'f_opt': [], 'dist_to_origin': [], 'iter_to_convergence': [], 'dist_moved_towards_origin': []}
    sd_ebls_results = {'dist_from_start': [], 'f_opt': [], 'dist_to_origin': [], 'iter_to_convergence': [], 'dist_moved_towards_origin': []}
    hball_results = {'dist_from_start': [], 'f_opt': [], 'dist_to_origin': [], 'iter_to_convergence': [], 'dist_moved_towards_origin': []}
    nesterov_results = {'dist_from_start': [], 'f_opt': [], 'dist_to_origin': [], 'iter_to_convergence': [], 'dist_moved_towards_origin': []}
    nesterov2_results = {'dist_from_start': [], 'f_opt': [], 'dist_to_origin': [], 'iter_to_convergence': [], 'dist_moved_towards_origin': []}

    for i in range(n_iter):
        if i%100 == 0:
            print("Random Restart: {}".format(i))
        x_0 = np.random.uniform(low=x_min, high=x_max, size=2)
        start_distance.append(np.linalg.norm(x_0))

        x_opt, f_opt, results_df = steepest_descent(x_0, alpha=alpha, use_ebls=False)
        save_results(sd_results, x_0, x_opt, f_opt, results_df.shape[0])

        x_opt, f_opt, results_df = steepest_descent(x_0, alpha=alpha, use_ebls=True)
        save_results(sd_ebls_results, x_0, x_opt, f_opt, results_df.shape[0])

        x_opt, f_opt, results_df = heavy_ball(x_0, alpha=alpha, beta=beta, use_ebls=False)
        save_results(hball_results, x_0, x_opt, f_opt, results_df.shape[0])

        x_opt, f_opt, results_df = nesterov(x_0, alpha=alpha, beta=beta, use_ebls=False)
        save_results(nesterov_results, x_0, x_opt, f_opt, results_df.shape[0])


    f_opt_df = pd.DataFrame()
    dist_to_origin_df = pd.DataFrame()
    dist_from_start_df = pd.DataFrame()
    dist_moved_towards_origin_df = pd.DataFrame()

    algo_results_map = {'sdesc': sd_results, 'sdesc_ebls': sd_ebls_results, 'heavy ball': hball_results, 'nestrov': nesterov_results, 'nestrov2': nesterov2_results}
    aggregated_results_df = {'algo': [], 'start_distance': [], 'dist_from_start': [], 'dist_to_origin': [],
                            'f_opt': [], 'iter_to_convergence': [], 'dist_moved_towards_origin': []}
    for algo in algo_results_map.keys():
        results = algo_results_map[algo]
        aggregated_results_df['algo'].append(algo)
        aggregated_results_df['start_distance'].append(np.mean(start_distance))
        aggregated_results_df['dist_from_start'].append(np.mean(results['dist_from_start']))
        aggregated_results_df['dist_to_origin'].append(np.mean(results['dist_to_origin']))
        aggregated_results_df['f_opt'].append(np.mean(results['f_opt']))
        aggregated_results_df['iter_to_convergence'].append(np.mean(results['iter_to_convergence']))
        aggregated_results_df['dist_moved_towards_origin'].append(np.mean(results['dist_moved_towards_origin']))

        f_opt_df[algo + '_f_opt'] = pd.Series(results['f_opt'])
        dist_to_origin_df[algo + '_d'] = pd.Series(results['dist_to_origin'])
        dist_from_start_df[algo + '_d'] = pd.Series(results['dist_from_start'])
        dist_moved_towards_origin_df[algo + '_d'] = pd.Series(results['dist_moved_towards_origin'])
        plot_aggregates(f_opt_df, dist_to_origin_df, dist_from_start_df, dist_moved_towards_origin_df)

    aggregated_results_df = pd.DataFrame(aggregated_results_df)
    print aggregated_results_df

    aggregated_results_df.to_csv("aggregated_results_{}.csv".format(datetime.datetime.now().strftime('%d_%m_%Y')))
    return aggregated_results_df