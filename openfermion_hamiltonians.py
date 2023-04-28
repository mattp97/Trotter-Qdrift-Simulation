import numpy as np
import os
from openfermion.hamiltonians import jellium_model
from openfermion.utils import Grid, count_qubits
from openfermion.transforms import jordan_wigner, get_fermion_operator
from openfermion.linalg import get_sparse_operator

from openfermion.chem import MolecularData
from openfermionpyscf import run_pyscf
from openfermion.chem import geometry_from_pubchem


FLOATING_POINT_PRECISION = 1e-10

def openfermion_matrix_list(qubit_operator):
    total_qubits = count_qubits(qubit_operator)
    matrix_list = []
    op_list = list(qubit_operator)
    for i in op_list:
        matrix_list.append(get_sparse_operator(i, total_qubits).toarray()) #changed from qubit operator and made no differnce
    return np.array(matrix_list)

    #Test -- shows ops are equivalent
def test_list_generator(openfermion_output):
    max_val = []
    of_generator = get_sparse_operator(openfermion_output).toarray()
    list_generator = sum(openfermion_matrix_list(openfermion_output))
    the_zero_op = of_generator - list_generator
    for i in range(the_zero_op.shape[0]):
        for j in range(the_zero_op.shape[0]):
            max_val.append((the_zero_op)[i][j])
    print(max(max_val))
    norm = np.linalg.norm(the_zero_op, ord=2)
    if norm < FLOATING_POINT_PRECISION:
        print("success!")
    else:
        print("failed!")
    return 0

def jellium_hamiltonian(dimensions, length, spinless=True):
    grid = Grid(dimensions=dimensions, length=length, scale=1.0)
    # Get the momentum Hamiltonian.
    momentum_hamiltonian = jellium_model(grid, spinless)
    momentum_qubit_operator = jordan_wigner(momentum_hamiltonian)
    momentum_qubit_operator.compress()

    #Generate the matrix list
    jellium_hamiltonian_list = openfermion_matrix_list(momentum_qubit_operator) #load this into simulator
    print("Hamiltonian has dimensions: " + str(jellium_hamiltonian_list.shape))
    #print(momentum_qubit_operator)
    #test_list_generator(momentum_qubit_operator)
    return jellium_hamiltonian_list

def hydrogen_chain_hamiltonian(chain_length, bond_length): 
    hydrogen_geometry = []
    for i in range(chain_length):
        hydrogen_geometry.append(('H', (bond_length * i, 0, 0)))

    #print("Geometry in use:")
    #print(hydrogen_geometry)
    basis = 'sto-3g'
    if chain_length % 2 == 0:
        multiplicity = 1 #2ns+1
    else:
        multiplicity = 2

    # Set Hamiltonian parameters.
    active_space_start = 0
    active_space_stop = chain_length 

    # Set calculation parameters (to populate the molecular data class)
    run_scf = False #Hartree-Fock
    run_mp2 = False #2nd order Moller-Plesset (special case of R-S PT)
    run_cisd = False # Configuration interaction with single and double excitations
    run_ccsd = False #Coupled Cluster
    run_fci = True #Full configuration interaction
    verbose = False

    # Generate and populate instance of MolecularData.
    hydrogen = MolecularData(hydrogen_geometry, basis, multiplicity, description="hydrogen_chain_" + str(chain_length) +"_"+str(bond_length), filename="hydrogen_" + str(chain_length) +"_"+str(bond_length))
    if os.path.exists(hydrogen.filename + '.hdf5'):
        hydrogen.load()
    else:
        hydrogen = run_pyscf(hydrogen, run_scf=run_scf, run_mp2=run_mp2, run_cisd=run_cisd, run_ccsd=run_ccsd, run_fci=run_fci, verbose=verbose)
        #two_body_integrals = hydrogen.two_body_integrals
        hydrogen.save()

    # Get the Hamiltonian in an active space.
    hydrogen_molecular_hamiltonian = hydrogen.get_molecular_hamiltonian(occupied_indices=range(active_space_start),
        active_indices=range(active_space_start, active_space_stop))

    # Map operator to fermions and qubits.
    hydrogen_fermion_hamiltonian = get_fermion_operator(hydrogen_molecular_hamiltonian)
    hydrogen_qubit_hamiltonian = jordan_wigner(hydrogen_fermion_hamiltonian)
    hydrogen_hamiltonian_list = openfermion_matrix_list(hydrogen_qubit_hamiltonian)
    #print(hydrogen_hamiltonian_list.shape)
    #print(hydrogen_hamiltonian_list)
    return hydrogen_hamiltonian_list