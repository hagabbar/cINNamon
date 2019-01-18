#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example of running emcee to fit the parameters of a straight line.
"""

from __future__ import print_function, division

import os
import sys
import numpy as np
import matplotlib as mpl
mpl.use("Agg") # force Matplotlib backend to Agg
import corner # import corner.py


# import emcee
import emcee

# import model and data
#from createdata import *

def logposterior(theta, data, sigma, x):
    """
    The natural logarithm of the joint posterior.
    
    Args:
        theta (tuple): a sample containing individual parameter values
        data (list): the set of data/observations
        sigma (float): the standard deviation of the data points
        x (list): the abscissa values at which the data/model is defined
    """
    
    lp = logprior(theta) # get the prior
    
    # if the prior is not finite return a probability of zero (log probability of -inf)
    if not np.isfinite(lp):
        return -np.inf
    
    # return the likeihood times the prior (log likelihood plus the log prior)
    return lp + loglikelihood(theta,data,sigma,x)


def loglikelihood(theta, data, sigma, x):
    """
    The natural logarithm of the joint likelihood.
    
    Args:
        theta (tuple): a sample containing individual parameter values
        data (list): the set of data/observations
        sigma (float): the standard deviation of the data points
        x (list): the abscissa values at which the data/model is defined
    
    Note:
        We do not include the normalisation constants (as discussed above).
    """
    
    # unpack the model parameters from the tuple
    m, c = theta
    w = 6.0*np.pi
    p = 1.0
    tau = 0.25
    
    # evaluate the model (assumes that the straight_line model is defined as above)
    #md = straight_line(x, m, c)
    
    # return the log likelihood
    #return -0.5*np.sum(((md - data)/sigma)**2)
    return -0.5*np.sum(((data - c*np.sin(w*x + p)*np.exp(-((x-m)/tau)**2))/sigma)**2)


def logprior(theta):
    """
    The natural logarithm of the prior probability.
    
    Args:
        theta (tuple): a sample containing individual parameter values
    
    Note:
        We can ignore the normalisations of the prior here.
    """
    
    lp = 0.
    
    # unpack the model parameters from the tuple
    m, c = theta
    
    # uniform prior on c
    cmin = 0. # lower range of prior
    cmax = 1.  # upper range of prior
    
    # set prior to 1 (log prior to 0) if in the range and zero (-inf) outside the range 
    lp = 0. if cmin < c < cmax else -np.inf
    
    # Gaussian prior on m
    mmu = 0.     # mean of the Gaussian prior
    msigma = 1. # standard deviation of the Gaussian prior
    #lp -= 0.5*((m - mmu)/msigma)**2
    lp = 0. if mmu < m < msigma else -np.inf
    
    return lp


def run(ydata,sigma,x,loglikelihood):
    Nens = 100   # number of ensemble points

    mmu = 0.     # mean of the Gaussian prior
    msigma = 1. # standard deviation of the Gaussian prior

    mini = np.random.uniform(mmu, msigma, Nens) # this is basically ydata

    cmin = 0.  # lower range of prior
    cmax = 1.   # upper range of prior

    cini = np.random.uniform(cmin, cmax, Nens) # initial c points

    inisamples = np.array([mini, cini]).T # initial samples

    ndims = inisamples.shape[1] # number of parameters/dimensions

    Nburnin = 25  # number of burn-in samples
    Nsamples = 25  # number of final posterior samples

    # set additional args for the posterior (the data, the noise std. dev., and the abscissa)
    argslist = (ydata, sigma, x)

    # set up the sampler
    sampler = emcee.EnsembleSampler(Nens, ndims, logposterior, args=argslist)

    # pass the initial samples and total number of samples required
    sampler.run_mcmc(inisamples, Nsamples+Nburnin);

    # extract the samples (removing the burn-in)
    postsamples = sampler.chain[:, Nburnin:, :].reshape((-1, ndims))

    # plot posterior samples (if corner.py is installed)
    #try:
    #except ImportError:
    #    sys.exit(1)

    print('Number of posterior samples is {}'.format(postsamples.shape[0]))
    #fig = corner.corner(postsamples, labels=[r"$m$", r"$c$"], truths=[0.5, 0.5])
    #fig.savefig('/home/hunter.gabbard/public_html/emcee.png')

    # first column is amplitude and second is t0

    return postsamples
