#!/usr/bin/python

#
# A model of the colour receptive field in macaque V1, reproducing data
# presented in Horwitz and Hass, 2012. 
#
# Ben Selby

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.cm as cm

fsize = 25

# Generates a 2-D Gabor function with the specified parameters
# Used for generating simple cell receptive fields
def gabor(sig_x, sig_y, theta, phi, k):
    
    sig_x = float(sig_x)/fsize
    sig_y = float(sig_y)/fsize
    vals = np.linspace(-np.pi, np.pi, fsize)
    xgrid,ygrid = np.meshgrid(vals,vals)
    xy = xgrid*np.cos(theta) + ygrid*np.sin(theta)
    the_sine = np.sin(k*xy + phi)
    the_gaussian = np.exp(-(xgrid/(2*sig_x**2))**2-(ygrid/(2*sig_y**2))**2)
    return the_gaussian*the_sine


def LIF_neuron(sig_x, sig_y, theta, phi, k, gain, J_bias, RC_factor):
    tau_ref = 0.002
    neuronRF = gabor(sig_x, sig_y, theta, phi, k, fsize)
    img
    
    J = sum( sum( neuronRF * img ) )
   
    
    return 


def main():
    
    rate = np.array( [10, 10, 15, 18, 20, 22, 16, 11, 6,  2, 0, 0, 0, 0, 10, 10, 15, 18, 20, 22, 16, 11, 6, 2, 0, 0, 0, 0] )
    orientation = np.array( [360, 354, 349, 345, 340, 335, 330, 326, 321, 318, 270, 240, 225, 200, 180, 174, 169, 165, 160, 155, 150, 146, 141, 138, 110, 90, 45, 20] )
    
    # Convert orientation to radians
    orientation = orientation * np.pi / 180
    
    sigx = 125
    sigy = 125
    theta = np.pi/6
    k = 1
    phi = 0
    
    popt, pconv = scipy.optimize.curvefit( LIF_neuron, orientation, rate )
    
    print "Neuron RF properties: " popt
    
    neuronRF = gabor( popt[0], popt[1], popt[2], popt[3], popt[4], fsize )
    
    plt.imshow( neuronRF, cm.gray )

    plt.show()

if __name__=="__main__":
    main()
