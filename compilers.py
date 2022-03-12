import numpy as np

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
    def __init__(self, hamiltonian_list = [], time=1.0, iterations=1, order = 1):
        self.hamiltonian_list = hamiltonian_list
        self.time = time
        self.iterations = iterations
        self.spectral_norms = []
        self.hilbert_dim = hamiltonian_list[0].shape()[0]
        self.compute_spectral_norms()

    # Helper function to compute spectral norm list. Used in constructor, isolated as function for
    # potential use besides internally. If used externally returns copy so it cannot be modified
    # externally
    def compute_spectral_norms(self):
        if self.spectral_norms.len() == 0:
            ret = [np.linalg.norm(h, ord=2) for h in hamiltonian_list]
            self.spectral_norms = ret 
        return self.spectral_norms.copy()

    # Computes and stores the total approximate unitary evolution operater e^{i H t} based on the
    # order and parameters as given.
    def generate_evolution_op(self):
        evol_op = np.identity(self.hilbert_dim)
        return evol_op
