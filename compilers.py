import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import math
from numpy import random
import cmath
import time
from sympy import S, symbols, printing

FLOATING_POINT_PRECISION = 1e-10

# A basic trotter simulator organizer.
# Inputs
# - hamiltonian_list: List of terms that compose your overall hamiltonian. Data type of each entry
#                     in the list is numpy matrix (preferably sparse, no guarantee on data struct
#                     actually used). Ex: H = A + B + C + D --> hamiltonian_list = [A, B, C, D]
#                     ASSUME SQUARE MATRIX INPUTS
# - time: Floating point (no size guarantees) representing time for TOTAL simulation, NOT per
#         iteration. "t" parameter in the literature.
# - iterations: "r" parameter. This object will handle repeating the channel r times and dividing 
#               overall simulation into t/r chunks.
# - order: The trotter order, represented as "2k" in literature. 
class TrotterSim:
    def __init__(self, hamiltonian_list = [], order = 1):
        self.hamiltonian_list = []
        self.spectral_norms = []
        self.hilbert_dim = hamiltonian_list[0].shape[0]
        self.order = order

        # Use the first computational basis state as the initial state until the user specifies.
        self.initial_state = np.zeros((self.hilbert_dim, 1))
        self.initial_state[0] = 1
        self.final_state = np.copy(self.initial_state)

        self.prep_hamiltonian_lists(hamiltonian_list)

    # Helper function to compute spectral norm list. Used solely constructor
    def prep_hamiltonian_lists(self, ham_list):
        for h in ham_list:
            temp_norm = np.linalg.norm(h, ord=2)
            if temp_norm < FLOATING_POINT_PRECISION:
                print("[prep_hamiltonian_lists] Spectral norm of a hamiltonian found to be 0")
                self.spectral_norms = []
                self.hamiltonian_list = []
                return 1
            self.spectral_norms.append(temp_norm)
            self.hamiltonian_list.append(h / temp_norm)
        return 0

    # Do some sanity checking before storing. Check if input is proper dimensions and an actual
    # quantum state.
    def set_initial_state(self, psi_init):
        global FLOATING_POINT_PRECISION
        if type(psi_init) != type(self.initial_state):
            print("[set_initial_state]: input type not numpy ndarray")
            return 1

        if psi_init.size != self.initial_state.size:
            print("[set_initial_state]: input size not matching")
            return 1

        # check that the frobenius aka l2 norm is 1
        if np.linalg.norm(psi_init, ord = 2) - 1.0 > FLOATING_POINT_PRECISION:
            print("[set_initial_state]: input is not properly normalized")
            return 1

        # check that each dimension has magnitude between 0 and 1
        for ix in range(len(psi_init)):
            if np.abs(psi_init[ix]) > 1.0:
                print("[set_initial_state]: too big of a dimension in vector")
                return 1

        # Should be good to go now
        self.initial_state = psi_init
        return 0

    # Helper functions to generate the sequence of gates for product formulas given an input time
    # up to the simulator function to handle iterations and such. Can probably move all of these 
    # into one single function.
    def first_order_op(self, op_time):
        evol_op = np.identity(self.hilbert_dim)
        for ix in range(len(self.hamiltonian_list)):
            h_term = self.hamiltonian_list[ix] * self.spectral_norms[ix]
            exp_h = linalg.expm(1.0j * op_time  * h_term)
            evol_op = np.matmul(evol_op, exp_h)
        return evol_op
    
    def second_order_op(self, op_time):
        forward = self.first_order_op(op_time / 2.0)
        backward = self.first_order_op(-1 * op_time / 2.0).conjugate().T
        return np.matmul(backward, forward)

    def higher_order_op(self, order, op_time):
        if type(order) != type(2):
            print("[higher_order_op] provided input order (" + str(order) + ") is not an integer")
            return 1
        elif order == 1:
            return self.first_order_op(op_time)
        elif order == 2:
            return self.second_order_op(op_time)
        elif order % 2 == 0:
            time_const = 1.0/(4 - 4**(1.0/order - 1))
            outer = np.linalg.matrix_power(self.higher_order_op(order - 2, time_const * op_time), 2)
            inner = self.higher_order_op(order - 2, (1. - 4. * time_const) * op_time)
            ret = np.matmul(outer, inner)
            ret = np.matmul(ret, outer)
            return ret
        else:
            print("[higher_order_op] Encountered incorrect order (" + str(order) + ") for trotter formula")
            return 1
    
    def simulate(self, time, iterations):
        if type(iterations) != type(3) or iterations < 1:
            print("[simulate] Incorrect type for iterations, must be integer greater than 1.")
            return 1
        evol_op = self.higher_order_op(self.order, (1.0 * time) / (1.0* iterations))
        evol_op = np.linalg.matrix_power(evol_op, iterations)
        self.final_state = np.dot(evol_op, self.initial_state)
        return np.copy(self.final_state)

    def infidelity(self, time, iterations):
        sim_state = self.simulate(time, iterations)
        good_state = np.dot(linalg.expm(1j * sum(self.hamiltonian_list) * time), self.initial_state)
        infidelity = 1 - (np.abs(np.dot(good_state.conj().T, sim_state)))**2
        return infidelity
    
    #def error_plot()
    
class QDriftSimulator:
    def __init__(self, hamiltonian_list = [], rng_seed = 1):
        self.hamiltonian_list = []
        self.spectral_norms = []
        self.hilbert_dim = hamiltonian_list[0].shape[0]
        self.rng_seed = rng_seed

        # Use the first computational basis state as the initial state until the user specifies.
        self.initial_state = np.zeros((self.hilbert_dim, 1))
        self.initial_state[0] = 1.
        self.final_state = np.copy(self.initial_state)

        self.prep_hamiltonian_lists(hamiltonian_list)
        np.random.seed(self.rng_seed)

    def prep_hamiltonian_lists(self, ham_list):
        for h in ham_list:
            temp_norm = np.linalg.norm(h, ord=2)
            if temp_norm < FLOATING_POINT_PRECISION:
                print("[prep_hamiltonian_lists] Spectral norm of a hamiltonian found to be 0")
                self.spectral_norms = []
                self.hamiltonian_list = []
                return 1
            self.spectral_norms.append(temp_norm)
            self.hamiltonian_list.append(h / temp_norm)
        return 0

    # Do some sanity checking before storing. Check if input is proper dimensions and an actual
    # quantum state.
    def set_initial_state(self, psi_init):
        global FLOATING_POINT_PRECISION
        if type(psi_init) != type(self.initial_state):
            print("[set_initial_state]: input type not numpy ndarray")
            return 1

        if psi_init.size != self.initial_state.size:
            print("[set_initial_state]: input size not matching")
            return 1

        # check that the frobenius aka l2 norm is 1
        if np.linalg.norm(psi_init, ord='fro') - 1.0 > FLOATING_POINT_PRECISION:
            print("[set_initial_state]: input is not properly normalized")
            return 1

        # check that each dimension has magnitude between 0 and 1
        for ix in range(len(psi_init)):
            if np.abs(psi_init[ix]) > 1.0:
                print("[set_initial_state]: too big of a dimension in vector")
                return 1

        # Should be good to go now
        self.initial_state = psi_init
        return 0

    # RETURNS A 0 BASED INDEX TO BE USED IN CODE!!
    def draw_hamiltonian_sample(self):
        sample = np.random.random()
        tot = 0.
        lamb = np.sum(self.spectral_norms)
        for ix in range(len(self.spectral_norms)):
            if sample > tot and sample < tot + self.spectral_norms[ix] / lamb:
                return ix
            tot += self.spectral_norms[ix] / lamb
        return len(self.spectral_norms) - 1

    def simulate(self, time, samples):
        evol_op = np.identity(self.hilbert_dim)
        tau = time * np.sum(self.spectral_norms) / (samples * 1.0)
        for n in range(samples):
            ix = self.draw_hamiltonian_sample()
            exp_h = linalg.expm(1.j * tau * self.hamiltonian_list[ix])
            evol_op = np.matmul(exp_h, evol_op)
        self.final_state = np.dot(evol_op, self.initial_state)
        return np.copy(self.final_state)

    def sample_channel_inf(self, time, samples, mcsamples):
        sample_fidelity = []
        for s in range(mcsamples):
            sim_state = self.simulate(time, samples)
            good_state = np.dot(linalg.expm(1j * sum(self.hamiltonian_list) * time), self.initial_state)
            sample_fidelity.append((np.abs(np.dot(good_state.conj().T, sim_state)))**2)
        infidelity = 1 - sum(sample_fidelity) / mcsamples 
        return infidelity
    

#class CompositeSim:
    #def __init__(self, hamiltonian_list = [], rng_seed = 1, order = 1):
        
        
 



    
# Create a simple evolution operator, compare the difference with known result. Beware floating pt
# errors
# H = sigma_X
def test_first_order_op():
    sigma_x = np.array([[0,1],[1,0]])
    sim = TrotterSim([sigma_x], order=1)

def test_second_order_op():
    sigma_x = np.array([[0, 1], [1, 0]])
    sim = TrotterSim([sigma_x], order=2)

def test_higher_order_op():
    sigma_x = np.array([[0,1], [1,0]])
    sim = TrotterSim([sigma_x], order=6)

def test_trotter():
    hilb_dim = 16
    X = np.array([[0, 1],[1, 0]], dtype='complex')
    Y = np.array([[0, -1j], [1j, 0]], dtype='complex')
    Z = np.array([[1, 0], [0, -1]], dtype='complex')
    I = np.array([[1, 0], [0, 1]], dtype='complex')
    
    # h1 = np.random.random() * np.kron(X, X)
    # h2 = np.random.random() * np.kron(X, Y)
    # h3 = np.random.random() * np.kron(X, Z)
    # h4 = np.random.random() * np.kron(Y, Z)
    # h5 = np.random.random() * np.kron(Y, Y)
    # h1 = np.random.random() * np.kron(X, X)

    h1 = np.random.randn(hilb_dim, hilb_dim) + 1j * np.random.randn(hilb_dim, hilb_dim)
    h1 += h1.conjugate().T
    h2 = np.random.randn(hilb_dim, hilb_dim) + 1j * np.random.randn(hilb_dim, hilb_dim)
    h2 += h1.conjugate().T
    h3 = np.random.randn(hilb_dim, hilb_dim) + 1j * np.random.randn(hilb_dim, hilb_dim)
    h3 += h1.conjugate().T
    h4 = np.random.randn(hilb_dim, hilb_dim) + 1j * np.random.randn(hilb_dim, hilb_dim)
    h4 += h1.conjugate().T
    h5 = np.random.randn(hilb_dim, hilb_dim) + 1j * np.random.randn(hilb_dim, hilb_dim)
    h5 += h1.conjugate().T
    h6 = np.random.randn(hilb_dim, hilb_dim) + 1j * np.random.randn(hilb_dim, hilb_dim)
    h6 += h1.conjugate().T

    h = [h1, h2, h3, h4, h5, h6]
    input_state = np.array([1] + [0] * (hilb_dim - 1), dtype='complex').flatten()

    sim1 = TrotterSim(h, order = 1)
    sim1.set_initial_state(input_state)
    sim2 = TrotterSim(h, order = 2)
    sim2.set_initial_state(input_state)
    sim4 = TrotterSim(h, order = 4)
    sim4.set_initial_state(input_state)

    iterations = 2**16
    t_list = np.logspace(-4, -1, 100)
    inf1 = []
    inf2 = []
    inf4 = []
    for t in t_list:
        inf1.append(sim1.infidelity(t, iterations))
        inf2.append(sim2.infidelity(t, iterations))
        inf4.append(sim4.infidelity(t, iterations))
    log_inf1 = np.log10(inf1).flatten()
    log_inf2 = np.log10(inf2).flatten()
    log_inf4 = np.log10(inf4).flatten()
    log_t = np.log10(t_list)

    # Note we use the same t scale for all orders
    fig, axs = plt.subplots(3, sharex = True)
    fig.suptitle("Log-log t vs infidelity for Trotter formula orders 1, 2, and 4")

    # Order 1
    axs[0].set_title("Order 1")
    axs[0].plot(log_t, log_inf1, 'bo-')
    axs[0].set(ylabel='log(infidelity)')

    fit_points = 50 # declare the starting point to fit in the data
    p = np.polyfit(log_t[0 : fit_points], log_inf1[0 : fit_points], 1)
    f = np.poly1d(p)

    t_new = np.linspace(log_t[fit_points], log_t[-1], 50)
    y_new = f(t_new)

    data = symbols("t")
    poly = sum(S("{:6.2f}".format(v)) * data**i for i, v in enumerate(p[::-1]))
    eq_latex = printing.latex(poly)

    axs[0].plot(t_new, y_new, 'r--', label = "${}$".format(eq_latex))
    axs[0].legend(fontsize = "large")

    # Order 2
    axs[1].set_title("Order 2")
    axs[1].plot(log_t, log_inf2, 'bo-')
    axs[1].set(ylabel='log(infidelity)')

    fit_points = 50 # declare the starting point to fit in the data
    p = np.polyfit(log_t[0 : fit_points], log_inf2[0 : fit_points], 1)
    f = np.poly1d(p)

    t_new = np.linspace(log_t[fit_points], log_t[-1], 50)
    y_new = f(t_new)

    data = symbols("t")
    poly = sum(S("{:6.2f}".format(v)) * data**i for i, v in enumerate(p[::-1]))
    eq_latex = printing.latex(poly)

    axs[1].plot(t_new, y_new, 'r--', label = "${}$".format(eq_latex))
    axs[1].legend(fontsize = "large")

    # Order 4
    axs[2].set_title("Order 4")
    axs[2].plot(log_t, log_inf4, 'bo-')
    axs[2].set(ylabel='log(infidelity)')

    fit_points = 50 # declare the starting point to fit in the data
    p = np.polyfit(log_t[0 : fit_points], log_inf4[0 : fit_points], 1)
    f = np.poly1d(p)

    t_new = np.linspace(log_t[fit_points], log_t[-1], 50)
    y_new = f(t_new)

    data = symbols("t")
    poly = sum(S("{:6.2f}".format(v)) * data**i for i, v in enumerate(p[::-1]))
    eq_latex = printing.latex(poly)

    axs[2].plot(t_new, y_new, 'r--', label = "${}$".format(eq_latex))
    axs[2].legend(fontsize = "large")
    plt.show()


def test_qdrift():
    time = 0.3
    bigN = 500
    X = np.array([[0, 1],[1, 0]], dtype='complex')
    Y = np.array([[0, -1j], [1j, 0]], dtype='complex')
    Z = np.array([[1, 0], [0, -1]], dtype='complex')
    I = np.array([[1, 0], [0, 1]], dtype='complex')
    
    h1 = np.kron(X, X)
    h2 = np.kron(X, Y)
    h3 = np.kron(X, Z)
    h4 = np.kron(Y, Z)
    h5 = np.kron(Y, Y)
    h = [h1, h2, h3, h4, h5]

    input_state = np.array([1, 0, 0, 0]).reshape((4,1))
    qdsim = QDriftSimulator(h)
    qdsim.set_initial_state(input_state)

    exact_op = linalg.expm(1j * sum(h) * time)
    expected = np.dot(exact_op, input_state)
    
    fidelities = []
    num_samps = 50
    print("here we go")
    for ix in range(50):
        qd_out = qdsim.simulate(time, bigN)
        tmp = np.abs(np.dot(expected.conj().T, qd_out))**2
        fidelities.append(tmp)
    print("[test_qd] empirical infidelity: ", 1 - sum(fidelities) / (1. * num_samps))

test = False
if test:
    test_first_order_op()
    test_second_order_op()
    test_higher_order_op()
    test_trotter()
    # test_qdrift()
