import numpy as np
from openfermion.hamiltonians import jellium_model
from openfermion.utils import Grid, count_qubits
from openfermion.transforms import jordan_wigner, fourier_transform, get_fermion_operator
from openfermion.linalg import eigenspectrum, qubit_operator_sparse, get_sparse_operator
from openfermion.ops import QubitOperator


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