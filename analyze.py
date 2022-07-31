#!/usr/bin/env python

from utils import *
from compilers import *
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

# Analyze the results from the results.pickle file of a previous run
def analyze_entry_point():
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
    else:
        print("[analyze_entry_point] No results.pickle file path provided. quitting.")
        sys.exit()
    try:
        results = pickle.load(open(base_dir + '/' + LAUNCHPAD + '/output/results.pickle', 'rb'))
    except:
        print("[analyze_entry_point] no results were found at: ", base_dir + '/' + LAUNCHPAD + '/output/results.pickle')
        sys.exit()
    print("results:")
    print(results)
    times = results["times"]
    for k,v in results.items():
        if k != "times":
            plt.plot(times, v, label=k)
    plt.show()

if __name__ == "__main__":
    analyze_entry_point()
