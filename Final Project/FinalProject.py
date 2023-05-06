import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math
from warnings import catch_warnings
from warnings import simplefilter
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
#from skopt.utils import use_named_args


# Notes: https://paperswithcode.com/paper/practical-bayesian-optimization-of-machine
# On this website shows some code implimentations that might be helpful

# This was a paper used as reference throughout the project, though many of the pymc3 
# functions are not compatible with Visual Studios python 3.10. Still a helpful resource. 
# https://github.com/AM207-Study-Group/Bayesian-Optimization-with-pymc3/blob/master/AM207_Final%20Project_%20Bayesian%20Optimization%20of%20ML%20Algorithms.ipynb

# objective function for bayesian optimization example
def objective(x, noise=0.1):
    noise = np.random.normal(loc=0, scale=noise)
    return (x**2 * math.sin(2 * math.pi * x)**6.0) + noise
    #return (10*(x - 0.5)**5 +5*(x-0.25)**3 + x)


# surrogate or approximation for the objective function
def surrogate(model, X):
    # catch any warning generated when making a prediction
    with catch_warnings():
    # ignore generated warnings
        simplefilter("ignore")
    return model.predict(X, return_std=True)
 
# optimize the acquisition function by generating sample x points, and running them through the acquisition function. 
def opt_acquisition(X, y, model, mode = "PI", x_min = 0, x_max = 1):
    # random search, generate random samples
    Xsamples = np.random.random(100)
    Xsamples = Xsamples.reshape(len(Xsamples), 1)
    
    # Create a list of possible x values to score
    # NOTE: Choosing this implimentation causes the Expected Improvement (EI) aquisition function
    # to chose x points that are directly next to exisiting points. Creating a random list of X values
    # gives better point selection. 
    ##Xsamples = np.arange(x_min, x_max, (x_max-x_min)/100)
    ##Xsamples = Xsamples.reshape(len(Xsamples), 1)
    ##zeros = np.zeros((len(X), 1))

    #### DEBUG ONLY
    #### Check for ideal values of epsilon, impliment a optimizer for values of this explore/exploit epsilon value
    ##epsilon = np.arange(0, 3, 0.3)    
    ##
    ##for ep in epsilon:
    ##    # calculate the acquisition function for each sample
    ##    scores = acquisition(X, Xsamples, model, mode = mode, epsilon = ep)
    ##
    ##    # locate the index of the largest scores
    ##    ix = np.argmax(scores)
    ##
    ##    plt.title(str(ep))
    ##    plt.plot(Xsamples, scores, linewidth = 2.0)
    ##    #plt.scatter(X,y)
    ##    #plt.scatter(Xsamples[ix], objective(Xsamples[ix], noise = 0), c="r")
    ##    plt.scatter(X,zeros)
    ##    plt.scatter(Xsamples[ix], 0, c="r")
    ##    plt.show()
    
    # Find and save the point with the greatest acquisition score based on specified method (mode). 
    scores = acquisition(X, Xsamples, model, mode = mode, epsilon = 0.50)
    ix = np.argmax(scores)

    return Xsamples[ix, 0]


# probability of improvement acquisition function (and others)
def acquisition(X, Xsamples, model, mode="EI", epsilon = 0.0):
    # calculate the best surrogate score found so far
    yhat, _ = surrogate(model, X)
    best = max(yhat)
    # calculate mean and stdev via surrogate function
    mu, std = surrogate(model, Xsamples)
    #mu = mu[:, 0]
    
    # Variable that helps determine exploration/exploitation
    #epsilon = 0.075

    if mode == "PI":
        # calculate the probability of improvement (PI)
        probs = stats.norm.cdf((mu - best - epsilon) / (std+1E-9))
        #probs = (mu - best-epsilon) / (std+1E-9)
    elif mode == "LCB":
        # calculate the Lower Confidence Bound (LCB)
        probs = stats.norm.cdf((mu - 2*std))
    else :
        # See Expected Improvement for derivation https://distill.pub/2020/bayesian-optimization/
        # This is slightly different than the one calculated in the paper
        # calculate the Expected Improvement (EI)
        diff = (mu - best - epsilon) / (std+1E-9)
        #probs = std* (stats.norm.cdf(diff)*diff + stats.norm.cdf(diff))
        probs = (std*diff*stats.norm.cdf(diff) + std* stats.norm.pdf(diff))
    return probs

# plot real observations vs model produced surrogate function
def plot(X, y, model, title):
    # plot the surrogate function
    # scatter plot of inputs and real objective function
    plt.scatter(X, y, c = 'b', marker = 'x')
    # line plot of surrogate function across domain
    Xsamples = np.asarray(np.arange(0, 1, 0.001))
    Xsamples = Xsamples.reshape(len(Xsamples), 1)
    ysamples, _ = surrogate(model, Xsamples)
    plt.plot(Xsamples, ysamples)
    # show the plot
    plt.title(title)
    plt.show()

# Code based off an online example for basic Bayesian Optimization
# https://machinelearningmastery.com/what-is-bayesian-optimization/
def bayes_optimization():

    # sample the domain sparsely with noise
    x_min = 0
    x_max = 1
    x_mid = (x_max+x_min)/2
    x_range = x_max-x_min
    X_orig = np.random.random(3)
    X = (X_orig*x_range)+x_min
    y = np.asarray([objective(x, noise = 0) for x in X])
    # reshape into rows and cols
    X = X.reshape(len(X), 1)
    y = y.reshape(len(y), 1)

    # Produce 100 random points of training data to score the model
    x_train = np.random.random(100)
    X_train = (x_train*x_range)+x_min
    y_train = np.asarray([objective(x, noise = 0) for x in X_train])
    X_train = X_train.reshape(len(X_train), 1)
    y_train = y_train.reshape(len(y_train), 1)
    
    ##### Define the list of hyperparameters we want to search
    #### NOTE: This is automatically done in the model with a defined Kernel
    ####search_space = list()
    ####search_space.append(Real(1e-6, 100.0, 'theta', name='theta'))
    ####search_space.append(Categorical(['linear', 'poly', 'rbf', 'sigmoid'], name='kernel'))
    ####search_space.append(Integer(1, 5, name='degree'))
    ####search_space.append(Real(1e-6, 100.0, 'log-uniform', name='gamma'))
    ####@use_named_args(search_space)

    # Define the Kernel as the 5/2 Matern. The lower limit is set below the default of 1e-05, which cuts down 
    # on total number of errors that appear during optimization
    # Look at this website for more information about covarience kernals:
    # https://www.mathworks.com/help/stats/kernel-covariance-function-options.html
    kernel = Matern(length_scale = 1.0, length_scale_bounds= (1e-09, 100000.0), nu=2.5)
    
    # define the model
    model = GaussianProcessRegressor(kernel = kernel, n_restarts_optimizer=2)
    #model = GaussianProcessRegressor()
    ##model.set_params(**params)

    # fit the model
    model.fit(X, y)

    ### Check some model parameters (for debugging only)
    ##theta_out = model.log_marginal_likelihood()
    ##kernel_params = model.kernel.get_params()
    ##kernel_theta1 = model.kernel.theta
    
    ## plot data points and early model prediction before hand
    #plot(X, y, model, "Plot the model before the fitting happens")
    x_org = np.copy(X)
    y_org = np.copy(y)
    current_score = list()

       
    # Generate a the data used for a plot of the final ground truth function
    x_plot = np.arange(x_min, x_max, x_range/1000)
    y_truth = np.asarray([objective(i, noise=0) for i in x_plot])

    # perform the optimization process
    for i in range(10):
        # select the next point to sample
        x = opt_acquisition(X, y, model, mode = "EI", x_min = x_min, x_max = x_max)
        # sample the point
        actual = objective(x, noise = 0)
        # summarize the finding
        est, _ = surrogate(model, [[x]])
        print('>x=%.3f, f()=%3f, actual=%.3f' % (x, est, actual))
        # add the data to the dataset
        X = np.vstack((X, [[x]]))
        y = np.vstack((y, [[actual]]))
        # update the model
        model.fit(X, y)
        theta_out = model.log_marginal_likelihood()
        #current_score.append(model.score(X_train, y_train))

        ###for i in range(len(y_truth)):
        ###    y_truth[i] = objective(x[i], noise = 0)
        ##plt.scatter(x_org, y_org, c = 'r')
        ##plt.plot(x_plot, y_truth, linewidth = 2.0)
        ##plot(X, y, model, "Plot Newly selected samples and the current model/surrogate function")


     
    kernel_theta2 = model.kernel.theta
    current_score.append(model.score(X_train, y_train))
    print('Current Score: %s' % (current_score))

    # plot all samples and the final surrogate function
    x = np.arange(x_min, x_max, x_range/1000)
    y_truth = np.asarray([objective(i, noise=0) for i in x])
    #for i in range(len(y_truth)):
    #    y_truth[i] = objective(x[i], noise = 0)
    plt.scatter(x_org, y_org, c = 'r')
    plt.plot(x, y_truth, linewidth = 2.0)
    plot(X, y, model, "Plot all samples and the final surrogate function")

# Run the optimization program. 
bayes_optimization()


