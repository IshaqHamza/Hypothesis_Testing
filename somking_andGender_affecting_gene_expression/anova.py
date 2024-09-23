# import necessary libraries
import numpy as np
from scipy.stats import f

# Following is a function to handle division by zero
@np.vectorize   # Decorator for the function to be vectorized for fast elementwise operations
def smart_ratio(x: float, y: float) -> float:
    '''Returns the ratio of x and y and handles pathogenic cases
    Assumes:
        0/0 = 0
        x/0 = infinity for x != 0'''
    if x == y:
        return 1
    if y == 0:
        return np.infty
    return x/y

# Main hypothesis test function
def anova_two_way(observations: np.array, groups, effect_vectors_null: list[list[int]], effect_vectors_alternate: list[list[int]], rank_null=None, rank_alternate=None) -> float:
    '''
    Performs multi-way ANOVA test on the given observations (target variables), returns the p-value.
    Effect vector of a data-point belonging to group C is a binary vector v such that mu_C = v^T (mu_f1,...,mu_fn).
    
    Parameters:-
        data: matrix with each column being observations of one instance of a group
        groups: array containing lists where all instances of group i occur in group[i]
        effect_vectors_null/effect_vectors_alternate: matrix containg effect vectors under null/alternate hypothesis of defferent groups as rows
        observations: The observations (samples from the distribution under question in the form of a column vector)
    '''

    # Make effect matrices
    group_sizes = [len(group) for group in groups]

    effect_matrix_null = np.array(sum(([effect_vector] * group_size for effect_vector, group_size in zip(effect_vectors_null, group_sizes)), []), dtype=int)
    effect_matrix_alternate = np.array(sum(([effect_vector] * group_size for effect_vector, group_size in zip(effect_vectors_alternate, group_sizes)), []), dtype=int)

    effect_matrix_null_T = effect_matrix_null.T
    effect_matrix_alternate_T = effect_matrix_alternate.T

    # Compute dimensions and ranks
    num_instances = observations.shape[1]

    if rank_null is None:
        rank_null = np.linalg.matrix_rank(effect_matrix_null)
    if rank_alternate is None:
        rank_alternate = np.linalg.matrix_rank(effect_matrix_alternate)


    # Compute the relavant sums of squared errors and degrees of freedom
    observations_T = observations.T

    # Numerator, X^T (I - N(N^T N)^dagger N^T) X, N = effect_matrix_null
    sum_squared_errors_null = np.einsum('ij,jk,ki->i', observations, np.eye(num_instances) - effect_matrix_null @ np.linalg.pinv(effect_matrix_null_T @ effect_matrix_null) @ effect_matrix_null_T, observations_T)
    degrees_freedom_null = rank_alternate - rank_null

    # Denominator, X^T (I - D(D^T D)^dagger D^T) X, D = effect_matrix_alternate
    sum_squared_errors_alternate = np.einsum('ij,jk,ki->i', observations, np.eye(num_instances) - effect_matrix_alternate @ np.linalg.pinv(effect_matrix_alternate_T @ effect_matrix_alternate) @ effect_matrix_alternate_T, observations_T, optimize='optimal')
    degrees_freedom_alternate = num_instances - rank_alternate

    # Compute the F-statistics f_stat = (dof_d/dof_n)((num)/(den) - 1)
    F_statistics = smart_ratio(degrees_freedom_alternate, degrees_freedom_null) * (smart_ratio(sum_squared_errors_null, sum_squared_errors_alternate) - 1)
    
    # return p-values
    p_val = f(degrees_freedom_null, degrees_freedom_alternate).sf
    return p_val(F_statistics)

if __name__ == 'main':
    pass