import numpy as np
import matplotlib.pyplot as plt
from scipy.special import betainc

def quadrant_count(x, y, x_data, y_data):
    na=nb=nc=nd=0.0 
    for i in range(len(x_data)):
        if y_data[i] > y:
            if x_data[i] > x:
                na = na + 1
            else:
                nb = nb + 1
        else:
            if x_data[i] > x:
                nd = nd + 1
            else:
                nc = nc + 1

    ff = 1./len(x_data)
    return ff*np.array([na,nb,nc,nd])

"""
Given two arrays, compute the correlation coefficient 'r' - the significance
level at which the null hypothesis of zero correlation is disproved, 'prob' -
whose small value indicates a significant correlation, and Fisher's 'z' whose
value can be used in further statistical tests - pg 638, Numerical Recipes in C 
"""
def pearsn(x_data, y_data):
    TINY = 1.0e-20
    ax = np.mean(x_data)
    ay = np.mean(y_data)
    # compute correlation coefficient:
    sxx=sxy=syy=0.
    for i in range(len(x_data)):
        xt = x_data[i]-ax
        yt = y_data[i]-ay
        sxx = sxx + xt**2
        syy = syy + yt**2
        sxy = sxy + xt*yt
    
    r = sxy / np.sqrt(sxx*syy + TINY)
    z = 0.5*np.log((1+r+TINY)/(1-r+TINY))
    df = len(x_data) - 2.
    t = r* np.sqrt(df/((1-r+TINY) * (1+r+TINY)))
    prob = betainc(0.5*df, 0.5, df/(df+t**2)) 
    
    return [r, prob, z]

def prob_ks(alam):
    a2 = -2.0*alam**2
    termbf = 0.0
    t_sum = 0.0
    fac = 2.0
    EPS1 = 0.01
    EPS2 = 1.0e-8
    for i in range(1,101):
        term = fac*np.exp(a2*i**2)
        t_sum = t_sum +term
        if np.abs(term) <= EPS1*termbf or np.abs(term) <= EPS2*t_sum:
            return t_sum
        fac = -1*fac
        termbf = np.abs(term)
    # if we fail to converge:
    return 1.0
    

# 2D Kolmogorov-Smirnov test for 2 samples 
# adapted from Numerical Recipes in C, pg. 645:
# "Small values of prob show that the two samples are significantly different.
# Note that the test is slightly disribution-dependent, so prob is only an estimate"
# Returns K-S statistic 'd' and its significance level 'prob'
def ks2d2s( x1, y1, x2, y2 ):
    print "Running a 2D, 2 sample Kolmogorov-Smirnov test!"
    
    if len(x1)!=len(y1) or len(x2)!=len(y2):
        raise ValueError('Data dimensions mismatch: x and y lengths should be the same.')
    
    d1 = 0.
    for i in range(len(x1)):
        quad_count1 = quadrant_count(x1[i],y1[i],x1,y1)
        quad_count2 = quadrant_count(x1[i],y1[i],x2,y2)
        d1 = np.fmax(d1, np.abs(quad_count1[0]-quad_count2[0]))
        d1 = np.fmax(d1, np.abs(quad_count1[1]-quad_count2[1]))
        d1 = np.fmax(d1, np.abs(quad_count1[2]-quad_count2[2]))
        d1 = np.fmax(d1, np.abs(quad_count1[3]-quad_count2[3]))
        
    d2 = 0.
    for i in range(len(x2)):
        quad_count1 = quadrant_count(x2[i],y2[i],x1,y1)
        quad_count2 = quadrant_count(x2[i],y2[i],x2,y2)
        d2 = np.fmax(d2, np.abs(quad_count1[0]-quad_count2[0]))
        d2 = np.fmax(d2, np.abs(quad_count1[1]-quad_count2[1]))
        d2 = np.fmax(d2, np.abs(quad_count1[2]-quad_count2[2]))
        d2 = np.fmax(d2, np.abs(quad_count1[3]-quad_count2[3]))
    
    d = (d1+d2)/2.
    
    sqen = np.sqrt(len(x1)*len(x2)/float(len(x1) + len(x2) ) )
    # get linear correlation coefficient for each sample:
    [r1,dum,dumm] = pearsn(x1,y1)
    [r2,dum,dumm] = pearsn(x2,y2)
    rr = np.sqrt(1.0 - 0.5*(r1**2 + r2**2))
    
    # estimate probability:
    prob = prob_ks(d*sqen/(1.0+rr*(0.25 - 0.75/sqen) ) )
    
    return d, prob
