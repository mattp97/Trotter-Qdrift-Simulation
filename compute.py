#!/usr/bin/env python

from utils import *
from compilers import *
from tests import *
import numpy as np
import sys
import time as time_this
import pickle
import os
import shutil
import matplotlib.pyplot as plt

def find_launchpad(base_dir):
    launchpad = base_dir + LAUNCHPAD + '/'
    if os.path.exists(launchpad + 'hamiltonian.pickle'):
        ham_path = launchpad + 'hamiltonian.pickle'
    else:
        ham_path = None
    if os.path.exists(launchpad + 'settings.pickle'):
        settings_path = launchpad + 'settings.pickle'
    else:
        settings_path = None
    return ham_path, settings_path

def get_base_dir():
    if len(sys.argv) > 1:
        base = sys.argv[1]
    else:
        print("[get_base_dir] No output directory given, using SCRATCH")
        scratch_path = os.getenv("SCRATCH")
        if scratch_path == '':
            print("[get_base_dir] nothing provided and no scratch set. bailing.")
            sys.exit()
        base = scratch_path
    if base[-1] != '/':
        base += '/'
    return base

def compute_entry_point():
    clock_start = time_this.time()
    base_dir = get_base_dir()
    ham_path, settings_path = find_launchpad(base_dir)
    if type(ham_path) != type("string") or type(settings_path) != type("string"):
        print("[tests.py] Error: could not find hamiltonian.pickle or settings.pickle")
        sys.exit()
    try:
        unprocessed = pickle.load(open(ham_path, 'rb'))
        shape = unprocessed.pop(-1)
        ham_list = [np.array(unprocessed[ix]).reshape(shape) for ix in range(len(unprocessed))]
        settings = pickle.load(open(settings_path, 'rb'))
    except:
        print("[compute_entry_point] you fool, we couldn't even unload the hamiltonian or settings")
        sys.exit()
    print("[compute_entry_point] Found hamiltonian at path: ", ham_path)
    print("[compute_entry_point] Contains this many terms: ", len(ham_list))
    print("[compute_entry_point] Found the following settings at file path: ", settings_path)
    for k,v in settings.items():
        print(k, "=", v)

    exp = Experiment(base_directory=base_dir, use_density_matrices=settings.get("use_density_matrices", False))
    exp.input_hamiltonian(ham_list)
    exp.input_settings(settings)
    exp.run()
    print("[compute] time taken:", time_this.time() - clock_start, " (sec)")

if __name__ == "__main__":
    compute_entry_point()
