#!/usr/bin/env python

from utils import *
from compilers import *
from tests import Experiment
import numpy as np
import sys
import pickle
import os
import shutil
import matplotlib.pyplot as plt

INFIDELITY_TEST_TYPE = "infidelity"
TRACE_DIST_TEST_TYPE = "trace_distance"
GATE_COST_TEST_TYPE = "gate_cost"
CROSSOVER_TEST_TYPE = "crossover"
LAUNCHPAD = "launchpad"
MINIMAL_SETTINGS = ["experiment_label",
                    "verbose",
                    'use_density_matrices',
                    't_start',
                    't_stop',
                    't_steps',
                    'partitions',
                    'infidelity_threshold',
                    'num_state_samples',
                    'output_directory',
                    'test_type'
]

def find_launchpad():
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
        if base_dir[-1] != '/':
            base_dir += '/'
    else:
        scratch_path = os.getenv("SCRATCH")
        if type(scratch_path) != type("string"):
            print("[find_launchpad] Error, could not find SCRATCH environment variable and no base directory provided")
            return None, None, None
        if scratch_path[-1] != '/':
            scratch_path += '/'
        base_dir = scratch_path
    launchpad = base_dir + LAUNCHPAD + '/'
    if os.path.exists(launchpad + 'hamiltonian.pickle'):
        ham_path = launchpad + 'hamiltonian.pickle'
    else:
        ham_path = None
    if os.path.exists(launchpad + 'settings.pickle'):
        settings_path = launchpad + 'settings.pickle'
    else:
        settings_path = None
    return ham_path, settings_path, launchpad

def compute_entry_point():
    ham_path, settings_path, launchpad = find_launchpad()
    if type(ham_path) != type("string") or type(settings_path) != type("string") or type(launchpad) != type("string"):
        print("[tests.py] Error: could not find hamiltonian.pickle or settings.pickle")
        sys.exit()
    ham_list = pickle.load(open(ham_path, 'rb'))
    settings = pickle.load(open(settings_path, 'rb'))
    print("[compute_entry_point] hamiltonian found with this many terms:", len(ham_list))
    print("#" * 50)
    print("settings found:")
    print(settings)
    working_dir = launchpad + 'output'
    if os.path.exists(working_dir) == False:
        os.mkdir(working_dir)
    exp = Experiment(output_directory=working_dir)
    exp.load_hamiltonian(ham_path)
    exp.load_settings(settings_path)
    exp.run()

if __name__ == "__main__":
    compute_entry_point()
