import math
import numpy
import scipy
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from scipy.special import erfinv
from scipy.optimize import minimize

# ===============================================
# 1st order polynomial pdf: y = intercept + slope * x
class Linear:
    
    # Simple constructor which sets no background, and 0-infinity limits
    def __init__(self, intercept, slope, lolim, hilim ):
        self.slope = 0
        self.intercept = 0
        self.lolimit = lolim
        self.hilimit = hilim
        self.mass = []
        self.shape = 0
        self.norm = 0
        self.renorm(intercept, slope)
        
    def renorm(self, intercept, slope):
        self.slope = slope
        self.intercept = intercept        
        self.shape = lambda t: self.intercept + self.slope * t
        self.norm = self.normalisation()
    
    # integral between xmin and xmax
    def integral(self, xmin, xmax):
        return integrate.quad(self.shape, xmin, xmax)[0]
    
    # Normalisaiton = the integral of the shape between the limits
    def normalisation(self):
        return self.integral(self.lolimit, self.hilimit)
    
    # Evaluate method (un-normalised)
    def evaluate(self, t):
        return self.shape(t)
    
    # Evaluate method (normalised)
    def evaluateNorm(self, t):
        return self.shape(t) / self.norm
    
    # Draw random value from distribution
    def next(self):
        doLoop = True
        while(doLoop):
            # start with uniform random number in [lolimit, hilimit)
            x = numpy.random.uniform(self.lolimit, self.hilimit)
            y1 = self.evaluate(x)
            y2 = numpy.random.uniform(0, self.maxval())
            if (y2 < y1):
                filtered_x =  x
                self.mass.append(filtered_x)
                return filtered_x

    # maximum value for linear (only works for negative slopes!)
    def maxval(self):
        return self.evaluate(self.lolimit)

#===============================================
# Gaussian pdf
class Gaussian:
    
    # constructor
    def __init__(self, mean, sigma):
        self.mean = mean
        self.sigma = sigma
        self.mass = []
        self.shape = lambda t: numpy.exp( -((t-self.mean)**2) / (2.0 * self.sigma**2 ) ) / (self.sigma*math.sqrt(2.0*math.pi ))
        self.norm = self.normalisation()
    
    # normalisation = the integral of the shape between lower & upper limits (here: fixed to a number of sigmas to keep things simple)
    def normalisation(self):
        num_of_sigmas = 6.0
        return self.integral(self.mean - num_of_sigmas * self.sigma, self.mean + num_of_sigmas * self.sigma)
    
    # integral between xmin and xmax
    def integral(self, xmin, xmax):
        return integrate.quad(self.shape, xmin, xmax)[0]
    
    # Evaluate method (un-normalised)
    def evaluate( self, t ):
        return self.shape(t)
    
    # Evaluate method (normalised)
    def evaluateNorm( self, t ):
        return self.shape(t) / self.norm
    
    # Draw random value from distribution
    def next(self):
        filtered_x =  numpy.random.normal(self.mean, self.sigma)
        self.mass.append(filtered_x)
        return filtered_x
    
    # maximum value for exponential (cannot generalise!)
    def maxval(self):
        return self.evaluate(self.mean)


#===============================================
# Linear + Gaussian pdf

class SignalWithBackground:
    
    # Simple constructor which sets no background, and 0-infinity limits
    def __init__(self, mean, sigma, sig_fraction, intercept, slope, lolim, hilim ):
        self.mean = mean
        self.sigma = sigma
        self.sig_fraction = sig_fraction
        self.lolimit = lolim
        self.hilimit = hilim
        self.signal = Gaussian(mean, sigma)
        self.background = Linear(intercept, slope, lolim, hilim)
        self.sig_fraction = sig_fraction
        self.shape_sig = lambda t: self.signal.evaluateNorm(t)
        self.shape_bgd = lambda t: self.background.evaluateNorm(t)
        self.reset()
    
    # Normalisaiton = the integral of the shape between the limits
    def normalisation(self):
        # since signal and background have different normalisations,
        # it is simpler to normalise the different contributions only once;
        # we do this in evaluate
        return 1
    
    # Evaluate method (un-normalised)
    def evaluate( self, t ):
        # Note that both component are normalised and that their weights add up to 1
        return self.sig_fraction * self.shape_sig(t) + (1 - self.sig_fraction) * self.shape_bgd(t)
    
    # Evaluate method (normalised)
    def evaluateNorm( self, t ):
        return self.evaluate(t) / self.normalisation()
    
    # reset the spectrum for every new experiment
    def reset(self):
        self.mass_sig = []
        self.mass_bgd = []
        self.mass = []
    
    # Draw random number form distribution
    def next(self):
        q = numpy.random.uniform()
        if( q < self.sig_fraction):
            # if here, we will draw x from signal distribution
            filtered_x = self.signal.next()
            self.mass_sig.append(filtered_x)
        else:
            # if here, we will draw x from background distribuion
            filtered_x = self.background.next()
            self.mass_bgd.append(filtered_x)
    
        self.mass.append(filtered_x)
        return filtered_x

#===============================================
# function to make x,y map of a shape
def mapShape( shape, lolimit, hilimit, steps ):
    y = []
    x = []
    increment = (hilimit-lolimit)/steps
    for i in range( steps ):
        t = lolimit+i*increment
        x.append(t)
        y.append(shape.evaluateNorm(t))
    return x,y
