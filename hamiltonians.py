#!/usr/bin/env python

import sys
import os
import numpy as np
import pickle
from utils import *

def pickle_hamiltonian(output_path, unparsed_ham_list):
    shape = unparsed_ham_list[0].shape
    # convert to pickle'able datatype
    to_pickle = [mat.tolist() for mat in unparsed_ham_list]
    to_pickle.append(shape)
    try:
        pickle.dump(to_pickle, open(output_path, 'wb'))
    except:
        print("[pickle_hamiltonian] could not pickle:" + output_path)
        return

def unpickle_hamiltonian(path):
    try:
        unparsed = pickle.load(open(path, 'rb'))
    except:
        print("[unpickle_hamiltonian] could not unpickle:", path)
        return
    shape = unparsed[-1]
    parsed = [np.array(unparsed[ix]).reshape(shape) for ix in range(len(unparsed) - 1)]
    return parsed
    

def hamiltonian_entry_point():
    scratch_path = os.getenv("SCRATCH")
    if type(scratch_path) != type("string"):
        print("[hamiltonian] use scratch path, currently not set")
        sys.exit()

    # if len(sys.argv) == 2:
        # ham_name = sys.argv[-1]
    # else:
        # print("[hamiltonian] give name")
        # sys.exit()

    if scratch_path[-1] != '/':
        scratch_path += '/'
    ham_path = scratch_path + 'hamiltonians'
    if os.path.exists(ham_path) == False:
        try:
            os.mkdir(ham_path)
        except:
            print("could not make hamiltonian folder and it does not exist.")
            sys.exit()
    for i in range(4,7):
        ham_list = graph_hamiltonian(i, 1, i)
        pickle_hamiltonian(ham_path + "/graph_" + str(i) + "1.pickle", ham_list)

if __name__ == "__main__":
    hamiltonian_entry_point()
