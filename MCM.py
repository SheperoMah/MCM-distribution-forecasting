#!/usr/bin/env python3
#
# MCM.py is a collection of functions for the Markov-chain
# Mixture Distribution model for forecasting. 
#
# The model was published in:
# [1] J. Munkhammar&J. Widén, Probabilistic forecasting of high-resolution 
# clear-sky index time-series using a Markov-chain mixture distribution 
# model, Solar Energy vol XX 2019. (Available as preprint on Researchgate)
#
# Any use of this model should cite the reference and state that this model
# was used.
#
# This file contains the following functions:
#
# P = MCMFit(Data,N) which determines an NxN transition matrix P from a 
# time-series in Data.
#
# X, Y = MCMForecast(P,a,b,obspoint) which delivers a piece-wise uniform
# distribution (X,Y) from the transition matrix P, minimum a and maximum b
# of the time-series and obspoint as the observation point to forecast
# from.
#
# NewSamples = MCMRnd(X,Y,Num) which delivers Num number of samples of the
# distribution X,Y (obtained in MCMForecast), which is the predictive 
# distribution (the forecast).
#
# These functions can all be tested using MCMtest.py, which is accompanied
# by a test data set TestData.txt

import numpy as np  

def MCMFit(data, n, timeSteps=1):
    """Estimates the transition probability to the future time-step.

    Parameters
    ----------
    data : (n,)
        Numpy array containing the data. This array should be of width 
        1.
    n : int
        Number of states to be fitted in the transition matrix.
    timeSteps : int, optional
        The time-steps of the returned transition matrix (the default
    is 1, which returns the transition matrix for the following time-
    step).
    
    Returns
    -------
    np.array(n,n)
       The transition matrix for the time-steps. 

    """
    # Set up bins and limits
    a = np.min(data)
    b = np.max(data)
    binWidth = (b - a) / n


    # Identify the states for the Markov-chain in the data set                    
    state = np.floor((data-a) / binWidth)
    state[state > (n-1)] = n - 1
    state = state.astype('int32')
    
    # Generate the transition matrix
    p = np.zeros((n,n))
    for i in range(1,len(data)):
        p[state[i-1], state[i]]= p[state[i-1], state[i]] + 1
 
    # Normalizing the transition matrix
    rowSums = p.sum(1)
    rowSums[rowSums == 0] = 1.0 # do not divide by zero
    p = p / rowSums[:, np.newaxis]

    p = np.linalg.matrix_power(p, timeSteps)    

    # Return the transition matrix
    return(p)
    

def MCMForecast(p, minValue, maxValue, obsPoint):
    """Returns the transition row and the bin stratring values.

    Parameters
    ---------
    p : np.arange(n,n)
        The transition matrix. 
    minValue : float
        The minimum value in the range of the data.
    maxValue : float
        The maximum value in the range of the data.
    obsPoint : float
        Observation from which to forecast.

    Returns
    -------
    np.array(n,)
        The bin starting values.
    np.array(n,)
        The row in the transition matrix p which represents starting
    from the observation.
     
    """
    assert minValue < maxValue, "minValue must be less than maxValue."
    assert obsPoint >= minValue, "Observation has to be larger than " \
    + "minValue."

    # The number of bins 
    n = p.shape[0]
   
    # Bin starting values 
    binWidth = (maxValue - minValue) / n 

    # Calculate the range of the bins
    binStarts = np.arange(n) * binWidth + minValue

    # Identify which bin the obspoint belongs to
    obsBin = np.where(obsPoint >= binStarts)[0][-1]  

    # Return the X and Y of the piece-wise uniform distribution
    return(binStarts, p[obsBin, :])


def MCMRnd(binStarts, transProbs, count):
    """Generate random forecasts from a bin.
    
    Parameters
    ----------
    binStarts : np.array(n,)
        An array contains the bin starting values.
    transProbs : np.array(n,)
        The transition probabilities from the forecast point.
    count : int
        Number of forecast samples.

    Returns
    -------
    np.array(count,)
        An array of forecast samples

    """
    # Calculate the bin-width
    binWidth = np.diff(binStarts)[0]
    
    # Set N as the the matrix size
    n = binStarts.shape[0]

    # Define the CDF (for later use of inverse CDF)    
    probsCDF = np.cumsum(transProbs)
 
    # Calculating the Num samples from the distribution
    # First setting initial conditions and a randomizer
    randWithinECDF = np.random.uniform(0, 1, count)
    # Setting the uniform random variable in each bin
    randWithinBin = np.random.uniform(0, binWidth, count)


    fcstSamples = np.zeros(count)        
    # Sampling from the CDF and then obtaining the inverse CDF
    for i in range(count):
        binIndex = np.where(randWithinECDF[i] <= probsCDF)[0][0]
        fcstSamples[i] = binStarts[binIndex] + randWithinBin[i]

    # Return the samples                
    return(fcstSamples)
    
