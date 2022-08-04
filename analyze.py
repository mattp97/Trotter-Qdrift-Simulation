#!/usr/bin/env python

from utils import *
from compilers import *
from math import floor, ceil
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
        if os.path.exists(os.getenv("SCRATCH")):
            print("[analyze_entry_point] Using scratch directory")
            base_dir = os.getenv("SCRATCH")
        else:
            print("[analyze_entry_point] No directory given and no scratch path. i give up.")
            sys.exit()
    if base_dir[-1] != '/':
        base_dir += '/'
    if os.path.exists(base_dir + "outputs"):
        output_dir = base_dir + 'outputs/'
    else:
        output_dir = base_dir
    print("[analyze_entry_point] Output directory: ", output_dir)
    print("[analyze_entry_point] possible outputs to load: ")
    dir_options = os.listdir(output_dir)
    for ix in range(len(dir_options)):
        print("option (" + str(ix + 1) + ") = ", dir_options[ix])
    response = input("which file to use: ")
    try:
        index = int(response)
        filename = dir_options[index - 1]
    except:
        filename = response
    if filename[-len(".pickle"):] != ".pickle":
        filename += ".pickle"
    try:
        results = pickle.load(open(output_dir + filename, 'rb'))
    except:
        print("[analyze_entry_point] no results were found at: ", output_dir + filename)
        sys.exit()
    print("results:")
    print(results)
    times = results["times"]
    if results.get("test_type", "") != CROSSOVER_TEST_TYPE:
        for p in POSSIBLE_PARTITIONS:
            if p in results:
                x,y = zip(*results[p])
                print("partition: ", p)
                for ix in range(len(x)):
                    print("x, y: ", x[ix], ", ", y[ix])
                logx = np.log10(np.abs(x))
                logy = np.log10(np.abs(y))
                try:
                    coef = np.polyfit(logx, logy, 1)
                    print("[analyze] linear fit of first half coefficients:", coef)
                except Exception as e:
                    print("[analyze] could not do linear fit, e: ", e)
                plt.loglog(x,y, label = p)
        plt.legend()
        plt.xlabel("times")
        plt.ylabel(results.get("test_type", ""))
        plt.title(" ".join(results.get("experiment_label", "default_label").split("_")))
        plt.show()

if __name__ == "__main__":
    analyze_entry_point()
