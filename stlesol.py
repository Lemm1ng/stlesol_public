import scipy as scp
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def acf_polynomial(x, lam=10.):
    '''Returns autocorreletion function values
    Input:
    x -- np.ndarray <numpy.float64>
    lam -- scaling factor <numpy.float64>
    Output:
    acf -- acf values at a given points <numpy.ndarray>'''
    return (1 + lam ** 2 * (x) ** 2) ** (-1 / 4)


def acf_gaussian(x, lam=1.):
    '''Returns autocorreletion function values
    Input:
    x -- np.ndarray <numpy.float64>
    lam -- scaling factor <numpy.float64>
    Output:
    acf -- acf values at a given points <numpy.ndarray>'''
    return np.exp(-lam * x ** 2)


def acf_exponential(x, lam=10.):
    '''Returns autocorreletion function values
    Input:
    x -- np.ndarray <numpy.float64>
    lam -- scaling factor <numpy.float64>
    Output:
    acf -- acf values at a given points <numpy.ndarray>'''
    return np.exp(-lam * np.abs(x))


def generate_rv_corr(x, acf, dx=1.):
    '''Generates rv with the specidied autocorrelation function.
    x -- points at which the rv is generated np.ndarray <numpy.float64>
    acf_function -- specified autocorrelation function
    dx -- coordinate step value <numpy.float64>'''

    rv_normal_real_space = np.random.normal(0., 1., x.shape[0]) / dx ** 0.5
    acf_real_space = acf(x)
    # FFT
    rv_normal_fourier_space = dx * np.fft.rfft(rv_normal_real_space)
    acf_fourier_space = dx * np.fft.rfft(acf_real_space)
    target_rv_fourier_space = (acf_fourier_space ** 0.5 * rv_normal_fourier_space)
    # IFFT
    target_rv = 1 / dx * np.fft.irfft(target_rv_fourier_space)

    return (target_rv)


def create_potential_evaluation_function(x, rv_x):
    '''Creates function which returns potential value at a given points based on the parabolic spline interpolation.
    Input:
    t -- points at which the potential is defined <numpy.ndarray>
    rv_t -- values of known potential values <numpy.ndarray>
    Output:
    evaluate_potential -- <function>'''
    # Adding the upper limit
    rv_x = np.hstack((rv_x, rv_x[0]))
    x = np.hstack((x, -x[0]))
    # print(rv_x[-1], x[-1], rv_x[0], x[0])
    evaluate_potential = interp1d(x, rv_x, kind='quadratic')
    # TBD: Periodic boundary conditions
    return evaluate_potential


def generate_alpha_rv(alpha, size=None):
    '''Generates random value from alpha-stable distribution
    Input:
    alpha -- distribution parameter <numpy.float64>
    size -- Output shape <tuple>. Single value returned if not specified.
    Output:
    rv_alpha -- <numpy.ndarray>'''

    W = -np.log(np.random.random(size=size))
    F = (0.5 - np.random.random(size=size)) * np.pi
    rv_alpha = (np.sin(alpha * F) / np.cos(F) ** (1 / alpha) *
                (np.cos((1 - alpha) * F) / W) ** ((1 - alpha) / alpha))
    return rv_alpha


def solve_le_corr_alpha_euler_periodic(dt, dx, t_steps, x_lim, acf_function, n_attempts=10,
                                       alpha=1.5, U0=1., K_alpha=1.):
    '''Solves the overdamped Langevin Equation with for the correlated potential
    and alpha-stable forces using the stochastic Euler method. Periodic boundary conditions are applied in space.
    Input:
    dt -- time step. It is used for the time derivative approximation. <float>
    dx -- coordinate step. It is used for the space derivative approximation. <float>
    t_steps -- Number of time steps for the integration <int>
    acf_function -- specified autocorrelation function for the potential field <function>
    n_attempts -- number of realizations <int>
    alpha -- parameter of the alpha-stable distribution <float>
    U0 -- amplitude of the random potential <float>
    K_alpha -- amplitude of the stochastic force <float>
    Output:
    x_sol -- solution statistics 2D <numpy.array>'''

    #  Generation of stochastic forses from alpha-stable distribution
    F_alpha = generate_alpha_rv(alpha, size=(t_steps + 1, n_attempts))

    #  Generation of stochastic potential
    x = np.arange(-x_lim, x_lim, dx)
    u_x = generate_rv_corr(x, acf_function, dx=dx)
    evaluate_potential = create_potential_evaluation_function(x, u_x)

    #  Setting up the initial conditions
    x_sol = np.zeros((t_steps + 1, n_attempts))

    #  uniform distribution across the potential landscapes
    x_sol[0, :] = np.linspace(-x_lim, x_lim, n_attempts)
    #  Solving the overdamped Langevin Equation
    for ii in range(1, t_steps + 1):
        # Periodicity is accounted in F_potential evaluation
        F_potential = (-U0
                       * (evaluate_potential(-x_lim + (x_sol[ii - 1, :] + dx + x_lim) % (2 * x_lim))
                          - evaluate_potential(-x_lim + (x_sol[ii - 1, :] - dx + x_lim) % (2 * x_lim)))
                       / 2 / dx)
        x_sol[ii, :] = x_sol[ii - 1, :] - F_potential * dt + (K_alpha * dt) ** (1 / alpha) * F_alpha[ii, :]
    return x_sol


def solve_le_corr_alpha_euler(dt, dx, t_steps, x_lim, acf_function, n_attempts=10,
                              alpha=1.5, U0=1., K_alpha=1.):
    '''Solves the overdamped Langevin Equation with for the correlated potential
    and random forces from alpha-stable distribution using the stochastic Euler method.
    Input:
    dt -- time step. It is used for the time derivative approximation. <float>
    dx -- coordinate step. It is used for the space derivative approximation. <float>
    t_steps -- Number of time steps for the integration <int>
    acf_function -- specified autocorrelation function for the potential field <function>
    n_attempts -- number of realizations <int>
    alpha -- parameter of the alpha-stable distribution <float>
    U0 -- amplitude of the random potential <float>
    K_alpha -- amplitude of the stochastic force <float>
    Output:
    x_sol -- solution statistics 2D <numpy.array>'''

    #  Generation of stochastic forses from alpha-stable distribution
    F_alpha = generate_alpha_rv(alpha, size=(t_steps + 1, n_attempts))

    #  Generation of stochastic potential
    x = np.arange(-x_lim, x_lim, dx)
    u_x = generate_rv_corr(x, acf_function, dx=dx)
    evaluate_potential = create_potential_evaluation_function(x, u_x)

    #  Setting up the initial conditions
    x_sol = np.zeros((t_steps + 1, n_attempts))

    #  Solving the overdamped Langevin Equation
    for ii in range(1, t_steps + 1):
        F_potential = (-U0
                       * (evaluate_potential((x_sol[ii - 1, :] + dx)) - evaluate_potential((x_sol[ii - 1, :] - dx)))
                       / 2 / dx)
        x_sol[ii, :] = x_sol[ii - 1, :] - F_potential * dt + (K_alpha * dt) ** (1 / alpha) * F_alpha[ii, :]
    return x_sol


def calculate_eamsd(x_sol):
    '''Calculates ensamble-averaged mean square displacement
    Input:
    x_sol -- solution statistics 2D <numpy.array>
    Output:
    x_msd -- mean squared displacement <numpy.ndarray>
    '''
    # print(x_sol[0,:].shape)
    return ((x_sol - x_sol[0, :]) * (x_sol - x_sol[0, :])).mean(axis=-1)


def calculate_fpt(t_sol, x_sol, dx_barrier=1.):
    '''Returns first passage times based on solution of Langevin equation.
    Input:
    t_sol -- times at which the solution of LE was obtained <numpy.ndarray>
    x_sol -- solution statistics 2D <numpy.array>
    dx_barrier -- position of the barrier relative to the intial point <float>
    Output:
    t_fpt -- first passage time <numpy.ndarray>'''
    t_fpt = np.inf * np.ones(x_sol[0, :].shape)
    for ii in range(x_sol.shape[1]):
        x_barrier_realization = x_sol[0, ii] + dx_barrier
        for jj in range(x_sol.shape[0]):
            if x_sol[jj, ii] >= x_barrier_realization:
                t_fpt[ii] = t_sol[jj]
                break
    return t_fpt