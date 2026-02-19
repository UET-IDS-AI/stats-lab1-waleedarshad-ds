import numpy as np
import matplotlib.pyplot as plt
# Histogram____________________________________________________________

def normal_histogram(n):
    data = np.random.normal(0, 1, n)
    plt.hist(data, bins=10)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Histogram of Normal(0,1)")
    plt.show()
    return data


def uniform_histogram(n):
    data = np.random.uniform(0, 10, n)
    plt.hist(data, bins=10)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Histogram of Uniform(0,10)")
    plt.show()
    return data


def bernoulli_histogram(n):
    data = np.random.binomial(1, 0.5, n)
    plt.hist(data, bins=10)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Histogram of Bernoulli(0.5)")
    plt.show()
    return data

# Sample Mean & Variable__________________________________________________

def sample_mean(data):
    return np.mean(data)


def sample_variance(data):
    n = len(data)
    mean = np.mean(data)
    # n-1 in denominat
    return np.sum((data - mean) ** 2) / (n - 1)   

#  Order Statistics___________________________________________________
# Returns (min, max, median)__________________________________________


def order_statistics(data):
    data_sorted = np.sort(data)

    minimum = np.min(data_sorted)
    maximum = np.max(data_sorted)
    median = np.median(data_sorted)

# Using percentile __________________________________________________
    q1 = np.percentile(data_sorted, 25)
    q3 = np.percentile(data_sorted, 75)

    return (minimum, maximum, median, q1, q3)

# Sample Covariance (n-1 denominator)__________________________________


def sample_covariance(x, y):
    n = len(x)
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    return np.sum((x - mean_x) * (y - mean_y)) / (n - 1)



#  Covariance Matrix (2×2)__________________________________________


def covariance_matrix(x, y):
    var_x = sample_variance(x)
    var_y = sample_variance(y)
    cov_xy = sample_covariance(x, y)

    return np.array([
        [var_x, cov_xy],
        [cov_xy, var_y]
    ])

