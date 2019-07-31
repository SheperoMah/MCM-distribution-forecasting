
data <- read.table("TestData.txt")
timeSeries <- unlist(data)


# Number of bins
numBins <- 50

# Number of sampled points from predictive distribution
numSamples <- 2000

# Number of steps ahead forecast
numStepAhead <- 1

# Observed point, from which to forecast
observation <- 0.5

# Obtain the NxN transition matrix P from the data 
# and Num number of steps ahead
p <- mcmFit(timeSeries, numBins, numStepAhead)

# Obtain X and Y for the piecewise uniform distribution
forecastToolsList  <- mcmForecast(p, min(timeSeries), max(timeSeries), observation)

# Generate Num random numbers of samples from the forecasted distribution
forecastSamples <- mcmRnd(forecastToolsList$binStartingValues, forecastToolsList$transitionProbs, numSamples)



