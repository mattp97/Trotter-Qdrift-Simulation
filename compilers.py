import numpy as np
from scipy import linalg

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

    # Helper functions to generate the sequence of gates for product formulas given an input time
    # up to the simulator function to handle iterations and such. Can probably move all of these 
    # into one single function.
    def first_order_op(self, op_time):
        evol_op = np.identity(self.hilbert_dim)
        for h_term in self.hamiltonian_list:
            exp_h = linalg.expm(1.0j * op_time  * h_term)
            evol_op = np.matmul(evol_op, exp_h)
        return evol_op
    
    def second_order_op(self, op_time):
        evol_op = np.identity(self.hilbert_dim)
        # Forwards terms
        for ix in range(len(self.hamiltonian_list)):
            exp_h = linalg.expm(1.0j * op_time * self.hamiltonian_list[ix] / 2.0)
            evol_op = np.matmul(evol_op, exp_h)
        
        # Backwards terms
        for ix in range(len(self.hamiltonian_list)):
            exp_h = linalg.expm(1.0j * op_time * self.hamiltonian_list[- ix - 1] / 2.0)
            evol_op = np.matmul(evol_op, exp_h)
        return evol_op

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
        tau = time / (samples * np.sum(self.spectral_norms))
        for n in range(samples):
            ix = self.draw_hamiltonian_sample()
            exp_h = linalg.expm(1.j * tau * self.hamiltonian_list[ix])
            evol_op = np.matmul(exp_h, evol_op)
        # print('[qd_sim] evol_op:\n', evol_op)
        self.final_state = np.dot(evol_op, self.initial_state)
        return np.copy(self.final_state)

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
    time = 1
    iterations = 1000
    X = np.array([[0, 1],[1, 0]], dtype='complex')
    Y = np.array([[0, -1j], [1j, 0]], dtype='complex')
    Z = np.array([[1, 0], [0, -1]], dtype='complex')
    I = np.array([[1, 0], [0, 1]], dtype='complex')
    
    h1 = np.kron(X, X)
    h2 = np.kron(X, Y)
    h3 = np.kron(X, Z)
    h4 = np.kron(Y, Z)
    h5 = np.kron(Y, Y)
    h1 = np.random.random() * np.kron(X, X)

    h = [h1, h2, h3, h4, h5]
    input_state = np.array([1, 0, 0, 0], dtype='complex').reshape((4,1))

    sim = TrotterSim(h, order = 2)
    sim.set_initial_state(input_state)

    exact_op = linalg.expm(1.j * sum(h) * time)
    expected = np.dot(exact_op, input_state)
    infidelities = []
    out = sim.simulate(time, iterations)
    infidelities.append(1. - np.abs(np.dot(expected.conj().T, out))**2)
    # print("[test_trotter] infidelities: ", infidelities)

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
    test_qdrift()
