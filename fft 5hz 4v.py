"""
Steps for Fourier Transform: 
    1. Construst a time signal ( Call the txt file in my case )
    2. Use fft from numpy
    3. PLot
"""

from matplotlib import pyplot
from pylab import genfromtxt
import matplotlib.pyplot as plt
from scipy.fftpack import fftfreq, fft, rfft
from scipy.signal import find_peaks
from statistics import mean , stdev
import numpy as np
from math import log10



# Importing the txt file 
mat0 = genfromtxt("4v 3.txt")
   

time = mat0[:,0]/1000
force = mat0[:,3]

plt.plot(time, force)
pyplot.ylabel("Force (mN)")
pyplot.xlabel("Time (sec)")
plt.show()


npts=len(time)

FFT = abs(rfft(force))
freqs = fftfreq(npts, time[1]-time[0])

plt.plot(freqs,FFT)
plt.grid()
plt.xlabel("frequency")
plt.ylabel("amplitude")
plt.xlim(-1,1)
plt.show()


#_______________________________________________________________________

# Frequency domain representation

fourierTransform = np.fft.fft(force)/len(force)           # Normalize amplitude

fourierTransform = fourierTransform[range(int(len(force)/2))] # Exclude sampling frequency

# How many time points are needed i,e., Sampling Frequency

samplingFrequency   = 10;

tpCount     = len(force)

values      = np.arange(int(tpCount/2))

timePeriod  = tpCount/samplingFrequency

frequencies = values/timePeriod

 

 

plt.plot(frequencies, abs(fourierTransform))
plt.xlabel("frequency")
plt.ylabel("amplitude")

plt.grid()

plt.show()

