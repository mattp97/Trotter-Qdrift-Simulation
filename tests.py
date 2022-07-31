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

# For coordinating runs. 
class Experiment:
    def __init__(
        self,
        hamiltonian_list = [],
        output_directory="./",
        experiment_label="default",
        t_start=1e-3,
        t_stop=1e-1,
        t_steps=50,
        num_state_samples=5,
        partitions=["first_order_trotter", "qdrift"],
        test_type=INFIDELITY_TEST_TYPE,
        infidelity_threshold=0.05,
        use_density_matrices=False,
        verbose=False
        ):
        self.sim = CompositeSim(hamiltonian_list=hamiltonian_list, use_density_matrices=use_density_matrices, verbose=verbose)
        self.use_density_matrices = use_density_matrices
        self.times = np.geomspace(t_start, t_stop, t_steps)
        self.partitions = partitions
        self.test_type=test_type.lower()
        self.infidelity_threshold = infidelity_threshold
        self.verbose = verbose
        self.num_state_samples = num_state_samples
        if output_directory[-1] != '/':
            self.output_directory = output_directory + '/'
        else:
            self.output_directory = output_directory
        self.experiment_label = experiment_label
    
    
    # TODO: implement multithreading?
    # TODO: How to handle probabilistic partitionings?
    def run(self):
        results = {}
        results["times"] = np.copy(self.times).tolist()
        for partition in self.partitions:
            if self.verbose:
                print("[Experiment.run] evaluating partition type:", partition)
            outputs = []
            heuristic = -1
            partition_sim(self.sim, partition)
            for t in results["times"]:
                if self.verbose:
                    print("[Experiment.run] evaluating time:", t)
                out = 0
                for _ in range(self.num_state_samples):
                    self.sim.randomize_initial_state()
                    if self.test_type == "infidelity":
                        inf_temp, _ = single_infidelity_sample(self.sim, t)
                        out += inf_temp
                    elif self.test_type == "gate_cost":
                        cost, iters = find_optimal_cost(self.sim, t, self.infidelity_threshold, heuristic=heuristic, verbose=self.verbose)
                        heuristic = iters
                        out += cost
                outputs.append(out / self.num_state_samples)
            results[partition] = outputs
        self.results = results
        pickle.dump(self.results, open(self.output_directory + "results.pickle", 'wb'))

    # TODO: This isn't really necessary anymore with the setup scripts
    def pickle_settings(self, output_path):
        settings = {}
        settings["experiment_label"] = self.experiment_label
        settings["verbose"] = self.verbose
        settings["use_density_matrices"] = self.use_density_matrices
        settings["t_start"] = self.times[0]
        settings["t_stop"] = self.times[-1]
        settings["t_steps"] = len(self.times)
        settings["partitions"] = self.partitions
        settings["infidelity_threshold"] = self.infidelity_threshold
        settings["num_state_samples"] = self.num_state_samples
        settings["output_directory"] = self.output_directory
        settings["test_type"] = self.test_type
        pickle.dump(settings, open(output_path, 'wb'))

    # given an exact absoulute path to a settings file, loads them into the experiment object. Note that we ensure defaults and formatting
    # are handled on write, so loading should not have to use these defaults but do this out of precaution. 
    def load_settings(self, settings_path):
        settings = pickle.load(open(settings_path, 'rb'))
        self.experiment_label     = settings.get('experiment_label', "default_experiment_label")
        self.verbose              = settings.get("verbose", True)
        self.use_density_matrices = settings.get("use_density_matrices", False)
        self.times                = np.geomspace(settings.get("t_start", 1e-4), settings.get("t_stop", 1e-2), settings.get("t_steps", 10))
        self.partitions           = settings.get("partitions", ["first_order_trotter", "qdrift"])
        self.infidelity_threshold = settings.get("infidelity_threshold", 0.05)
        self.num_state_samples    = settings.get("num_state_samples", 5)
        self.test_type            = settings.get("test_type", GATE_COST_TEST_TYPE)
        # Drop self.output_directory? if we can find the settings then just output there

    # Inputs:
    # - An absoluate path to a pickled hamiltonian file. converts to numpy array and makes sure simulator is loaded with that hamiltonian.
    def load_hamiltonian(self, input_path):
        unpickled = pickle.load(open(input_path, 'rb'))
        output_shape = unpickled[-1]
        ham_list = []
        for ix in range(len(unpickled) - 1):
            ham_list.append(np.array(unpickled[ix]).reshape(output_shape))
        print("[load_hamiltonian] loaded this many terms: ", len(ham_list))
        self.sim.set_hamiltonian(ham_list)

def pickle_hamiltonian(output_path, unparsed_ham_list):
    shape = unparsed_ham_list[0].shape
    # convert to pickle'able datatype
    to_pickle = [mat.tolist() for mat in unparsed_ham_list]
    to_pickle.append(shape)
    pickle.dump(to_pickle, open(output_path, 'wb'))

def hamiltonian_entry_point():
    if len(sys.argv) == 3:
        ham_path = sys.argv[2]
    else:
        print("[hamiltonian_entry_point] No hamiltonian store path provided. quitting.")
        sys.exit()
    graph_path = ham_path + "/graph_4_2_1.pickle"
    if os.path.exists(graph_path):
        print("[hamiltonian_entry_point] graph_4_2_1.pickle exists")
        sys.exit()
    else:
        print("[hamiltonian_entry_point] no file found, using this as file path: ", graph_path)
    ham_list = graph_hamiltonian(4,2,1)
    pickle_hamiltonian(graph_path, ham_list)
