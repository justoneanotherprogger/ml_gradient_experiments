from sympy import Matrix, diff
from utils import calculate_mse, get_dict_from_vectors


def gradient_downside(gradient_vector, mse_function, variables, initial_point=Matrix((0, 0)), precision=0.01):
    subs_for_grad = get_dict_from_vectors(variables, initial_point)
    old_mse = calculate_mse(mse_function, subs_for_grad)
    # print(old_mse)

    grad = Matrix([gradient_vector[i].subs(subs_for_grad) for i in range(len(gradient_vector))])
    new_point = initial_point - precision * grad
    subs_for_grad = get_dict_from_vectors(variables, new_point)
    new_mse = calculate_mse(mse_function, subs_for_grad)
    # print(new_mse)

    if old_mse / new_mse > 1 + precision:
        new_point = gradient_downside(gradient_vector, mse_function, variables, new_point, precision)
    # else:
    #     print(old_mse / new_mse)
    
    return new_point

def get_gradient_vector(function, variables):
    return Matrix([diff(function, variable) for variable in variables])