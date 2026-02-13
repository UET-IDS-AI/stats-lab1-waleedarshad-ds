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

# Example Test Run_________________________________________________

if __name__ == "__main__":

    # Q1 Example:::::::::::::::::::::::::::::::
    data1 = normal_histogram(1000)
    data2 = uniform_histogram(1000)
    data3 = bernoulli_histogram(1000)

    # Q2:::::::::::::::::::::::::::::::::::::
    print("Mean:", sample_mean(data1))
    print("Variance:", sample_variance(data1))

    # Q3:::::::::::::::::::::::::::::::::::::
    test_data = np.array([5,1,3,2,4])
    print("Order Stats (min, max, median, Q1, Q3):", order_statistics(test_data))

    # Q4:::::::::::::::::::::::::::::::::::::
    x = np.array([1,2,3,4,5])
    y = np.array([2,4,6,8,10])
    print("Sample Covariance:", sample_covariance(x, y))

    # Q5::::::::::::::::::::::::::::::::::::::
    print("Covariance Matrix:\n", covariance_matrix(x, y))
