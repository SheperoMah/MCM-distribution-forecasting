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

def MCMFit(Data,N,Power):

    # Set up bins and limits
    a = np.min(Data)
    b = np.max(Data)
    binWidth = (b - a) / N

    # Define the bins
    Bin = np.zeros(N)
    for i in range(0,N):
        Bin[i] = ((i)/N)*(b-a)

    # Identify the states for the Markov-chain in the data set                    
    State = np.floor((Data-a)*(N)/(b-a))
    State[State>(N-1)] = N-1
    
    # Generate the transition matrix
    P = np.zeros((N,N))
    for i in range(1,len(Data)):
        P[int(State[i-1]),int(State[i])]= P[int(State[i-1]),int(State[i])]+1
 
    # Normalizing the transition matrix
    for i in range(0,N):
        P[i,:] = P[i,:]/sum(P[i,:])

    # Failsafe, in case of NANs
    P[np.isnan(P)] = 0

    P = np.linalg.matrix_power(P, Power)    

    # Return the transition matrix
    return(P)
    
def MCMForecast(P,a,b,obspoint):
    assert a < b, "a must be less than b."

    # The number of bins 
    N = P.shape[0]
    
    # Identify which bin the obspoint belongs to
    for i in range(0,N):
        if obspoint>b:
            obsbin = N-1
        if obspoint>=a+abs(b-a)*(i)/N and obspoint<a+abs(b-a)*(i+1)/N :
            obsbin = i
           
    # Calculate the range of the bins
    Bins = np.zeros(N)
    for i in range(0,N):
        Bins[i] = a+abs(b-a)*(i)/N

    # Return the X and Y of the piece-wise uniform distribution
    return(Bins,P[obsbin,np.arange(0,N)])

def MCMRnd(P,X,Y,Num):
    import numpy as np  
    # Calculate the bin-width
    diff = abs(X[2]-X[1])
    
    # Set N as the the matrix size
    N = np.shape(X)[0]

    # Define the CDF (for later use of inverse CDF)    
    Prow = np.zeros(N+1)
    for i in range(0,N+1):
        Prow[i] = sum(Y[np.arange(0,i)])
    
    # Calculating the Num samples from the distribution
    # First setting initial conditions and a randomizer
    NewSamples = np.zeros(Num)        
    r = np.random.uniform(0,1,Num)
    # Setting the uniform random variable in each bin
    r2 = np.random.uniform(0,1,Num)
    # Sampling from the CDF and then obtaining the inverse CDF
    for i in range(0,Num):
        for j in range(0,N+1):
            if r[i]>=Prow[j] and r[i]<Prow[j+1]:
                NewSamples[i] = X[j]+diff*r2[i]

    # Return the samples                
    return NewSamples
    

    
    
    