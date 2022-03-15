import numpy as np
from sklearn.linear_model import Ridge, Lasso

# This file holds all the different least squares functions that are used in the package.
# All functions return the the least squares solution and the sun of squared residuals
# the *_manual functions are not used and are internal. They serve as an intuition for the faster package-delivered
# solutions, since they do the same, but it takes longer to perform

def least_squares(a_mat, b, regularization='None', reg_lambda=0):
    """
    :param a_mat: Matrix with dimensions (M, N)
    :param b: Vector of dimensions (M,)
    :param regularization: type of regularization to apply:
            - 'None' - Ordinary Least Squares (OLS) solving without regularization
            - 'ridge' - Applying ridge regression, L2 regularization
            - 'lasso' - Applying lasso regression, L1 regularization
    :param reg_lambda: the lambda factor for the regularization
    :return: Solving Ax = b MMSE estimation with different regularization types
    """
    if regularization == 'None':
        return ols(a_mat, b)
    elif regularization == 'ridge':
        return ridge_regression(a_mat, b, reg_lambda)
    elif regularization == 'lasso':
        return lasso_regression(a_mat, b, reg_lambda)
    else:
        raise ValueError('Regularization type not supported')


def ols(a_mat, b):
    """
    :param a_mat:
    :param b:
    :return: Ordinary Least Squares, numpy implementation
    """
    ls_result = np.linalg.lstsq(a_mat, b, rcond=-1)
    return ls_result[0], ls_result[1]

def ridge_regression(a_mat, b, reg_lambda):
    """
    :param a_mat:
    :param b:
    :param reg_lambda:
    :return: performing least squares with L2 regularization, sklearn implementation
             ||y - Xw||^2_2 + reg_lambda * ||w||^2_2
    """
    model = Ridge(alpha=reg_lambda)
    model.fit(a_mat, b)
    x_hat = model.coef_
    ssr   = np.sum(np.power(a_mat @ x_hat - b, 2))
    return x_hat, ssr

def lasso_regression(a_mat, b, reg_lambda):
    """
    :param a_mat:
    :param b:
    :param reg_lambda:
    :return: performing least squares with L1 regularization, sklearn implementation
             ||y - Xw||^2_2 + reg_lambda * ||w||_1
    """
    model = Lasso(alpha=reg_lambda)
    model.fit(a_mat, b)
    x_hat = model.coef_
    ssr   = np.sum(np.power(a_mat @ x_hat - b, 2))
    return x_hat, ssr

def ols_manual(a_mat, b):
    """
    :param a_mat:
    :param b:
    :return: ordinary least squares, manual computation, same as numpy function but slower
    """
    # x_hat = np.linalg.inv(a_mat.T @ a_mat) @ a_mat.T @ b
    x_hat = np.linalg.pinv(a_mat) @ b
    ssr = np.sum(np.power(a_mat @ x_hat - b, 2))
    return x_hat, ssr

def ridge_regression_manual(a_mat, b, reg_lambda):
    """
    :param a_mat:
    :param b:
    :param reg_lambda:
    :return: performing least squares with L2 regularization, manual computation, same as sklearn function but slower
    """
    x_hat = np.linalg.inv(a_mat.T @ a_mat + reg_lambda * np.eye(a_mat.shape[1])) @ a_mat.T @ b
    ssr = np.sum(np.power(a_mat @ x_hat - b, 2))
    return x_hat, ssr
