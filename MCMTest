#!/usr/bin/env python3
#
# MCMTest.py is a program for testing the Markov-chain Mixture Distribution
# (MCM) model for forecasting a time-series. It utilizes MCM.py as main model
# file and as a test data set TestData.m as input for forecasting.
#
# Import the MCM file
from MCM import MCMFit, MCMForecast, MCMRnd
import matplotlib.pyplot as plt
import numpy as np  

# Load the files
TestFilename = "TestData.txt"
Data = np.loadtxt(TestFilename).transpose()  

# Set the number of bins
N=50

# Number of sampled points from predictive distribution
Num = 2000

# Number of steps ahead forecast
Steps = 1

# Observed point, from which to forecast
Obspoint = 0.5

# Obtain the NxN transition matrix P from the data
# and Num number of steps ahead
P=MCMFit(Data,N,Steps)

# Obtain X and Y for the piecewise uniform distribution
X, Y  = MCMForecast(P,min(Data),max(Data), Obspoint)

# Generate Num random numbers of samples from the forecasted distribution
NewSamples = MCMRnd(P,X,Y,Num)

plt.hist(NewSamples,30)
plt.plot(X,Num*Y)
plt.axis([0.3, 0.8, 0, 800])
