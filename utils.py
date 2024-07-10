def get_mse_function(count, list_of_functions):
    total_result = 0
    for elem in list_of_functions:
        total_result += elem ** 2
    return total_result / count

def calculate_mse(function, subs):
    return function.subs(subs)

def get_dict_from_vectors(vector_1, vector_2):
    return {i: k for i, k in zip(vector_1, vector_2)}