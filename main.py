# My attempt to create an example of OLS Estimation -- Yi = alpha + betaXi + Epsiloni
# Procedure
# 1. Create data
# 2. Estimation model
# 3. Iteration
# 4. Visualize estimated coefficients in graphs

# import libraries
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# set seed for reproducibility
np.random.seed(1000)


def create_data(N, alpha, beta):
    """ Generate data for X, Epsilon, and Y variables respectively

    Args:
        N: number of samples
        alpha: intercept in Yi = alpha + betaXi + Epsiloni
        beta: coefficient of Xi in Yi = alpha + betaXi + Epsiloni

    Returns: generated dataframe for X, Epsilon, and Y variables

    """

    # generate data for X variable
    d = pd.DataFrame({'X': np.random.normal(loc=20, scale=4, size=N)})

    # generate data(np array) for Epsilon
    ep = np.random.normal(loc=0, scale=1, size=N)

    # generate data for Y
    d['Y'] = alpha + beta * d['X'] + ep

    return d


def est_model(N, alpha, beta):
    """ Estimate linear regression model

    Args:
        N: number of samples
        alpha: intercept in Yi = alpha + betaXi + Epsiloni
        beta: coefficient of Xi in Yi = alpha + betaXi + Epsiloni

    Returns: dataframe of estimated coefficients of X variable

    """

    # Generate data
    d = create_data(N, alpha, beta)

    # Use smf.ols to generate model estimates
    m = smf.ols('Y~X', data=d).fit()

    return m.params['X']


def iterate(N, alpha, beta, iters):
    """ Iterate simulation programs

    Args:
        N: number of samples
        alpha: intercept in Yi = alpha + betaXi + Epsiloni
        beta: coefficient of Xi in Yi = alpha + betaXi + Epsiloni
        iters: number of iterations

    Returns: dataframe of estimated coefficients of X

    """
    # generate beta_head list
    results = [est_model(N, alpha, beta) for i in range(0, iters)]

    # convert the list into dataframe
    df = pd.DataFrame(results, columns=['beta_head'])

    return df


df = iterate(2000, 10, 20, 200)
ax = df.plot.hist(column=['beta_head'],bins=12)
plt.show()


