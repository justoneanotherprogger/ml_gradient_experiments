from random import randint
import pandas as pd
from sympy import Matrix, Symbol
from gradient import get_gradient_vector, gradient_downside
from utils import get_dict_from_vectors, get_mse_function


def ml_studying(list_of_data_lists, list_of_results):
    matrix_slau = Matrix(list_of_data_lists)
    vector_results = Matrix(list_of_results)
    list_of_weights = Matrix([Symbol(f'x{i}') for i in range(matrix_slau.shape[1])])
    mse_function = get_mse_function(matrix_slau.shape[0], matrix_slau * list_of_weights - vector_results)
    # first_mse_value = calculate_mse(mse_function, get_dict_from_vectors(list_of_weights, (0, 0)))
    # print(first_mse_value)

    gradient_vector = get_gradient_vector(mse_function, list_of_weights)
    result = gradient_downside(gradient_vector, mse_function, list_of_weights)
    final_mse_value = mse_function.subs(get_dict_from_vectors(list_of_weights, result)) # type: ignore
    print(f"Result {tuple(result.T.tolist()[0])} MSE =", final_mse_value)

    final_result = (result.T * list_of_weights)[0]
    # print(final_result.subs(get_dict_from_vectors(list_of_weights, (1, 2))))
    return final_result


dataset = pd.DataFrame([[randint(-100, 100) for _ in range(3)] for _ in range(10)], columns=['number1', 'number2', 'target'])
print(dataset)

result_function = ml_studying(dataset.drop(columns='target'), dataset['target'])
print(result_function)