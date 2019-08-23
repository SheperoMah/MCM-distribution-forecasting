#!/usr/bin/env python3
#
# MCMTest.py is a program for testing the Markov-chain Mixture Distribution
# (MCM) model for forecasting a time-series. It utilizes MCM.py as main model
# file and as a test data set TestData.m as input for forecasting.
#
# Import the MCM file

import numpy as np  
import matplotlib.pyplot as plt
from MCM import MCMFit, MCMForecast, MCMRnd

# Load the files
testFilename = "TestData.txt"
data = np.loadtxt(testFilename).transpose()  

# Set the number of bins
n = 50

# Number of sampled points from predictive distribution
num = 2000

# Number of steps ahead forecast
steps = 1

# Observed point, from which to forecast
obsPoint = 0.5

# Obtain the NxN transition matrix P from the data
# and Num number of steps ahead
p = MCMFit(data, n, steps)

# Obtain X and Y for the piecewise uniform distribution
binStarts, transProbs = MCMForecast(p, data.min(), data.max(), obsPoint)

# Generate Num random numbers of samples from the forecasted distribution
fcstSamples = MCMRnd(binStarts, transProbs, num)

plt.hist(fcstSamples, 30)
plt.plot(binStarts, num*transProbs)
plt.axis([0.3, 0.8, 0, 800])
plt.show()
