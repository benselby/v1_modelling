#!/usr/bin/env python

###
# A mini library containing the functions typically used when running 
# simulations using the supralinear stabilized network (Rubin et al., 2015).
# 
# Ben Selby, September 2015

import numpy as np
import scipy.io
import matplotlib.image as mpimg

class SSNetwork:

    # Default constructor - use the Rubin et al. parameters to produce a SSN layer:
    def __init__(self, N_pairs=75, sig_EE=8, sig_IE=12, sig_EI=4, sig_II=4, J_EE=0.1, J_IE=0.38, J_EI=0.089, J_II=0.096, OP_map=0, OD_map=0):

        self.N_pairs = N_pairs # no. of E/I pairs to a side of a grid
        self.field_size = 16. # size of field to a side (degrees)
        self.dx = field_size / N_pairs

        self.sig_FF = 32.
        self.sig_RF = dx

        self.k   = np.random.normal(0.012, 0.05*0.012, (N_pairs, N_pairs))
        self.n_E = np.random.normal(2.0, 0.05*2.0, (N_pairs, N_pairs)) 
        self.n_I = np.random.normal(2.2, 0.05*2.2, (N_pairs, N_pairs))

        self.tau_E = np.random.normal(0.02, 0.05*0.02, (N_pairs, N_pairs))
        self.tau_I = np.random.normal(0.01, 0.05*0.01, (N_pairs, N_pairs))

        # Connection weight parameters (from supp. materials S1.1.2):
        self.kappa_E = 0.1
        self.kappa_I = 0.5

        # kappa_E = 0.18
        # kappa_I = .85

        self.J_EE = J_EE
        self.J_IE = J_IE
        self.J_EI = J_EI
        self.J_II = J_II

        self.sig_EE = sig_EE*dx
        self.sig_IE = sig_IE*dx
        self.sig_EI = sig_EI*dx
        self.sig_II = sig_II*dx
        self.sig_ori = 45.

        if OP_map == 0:
            try:
                # load OP map from Bryan's extracted Kaschube map
                data = scipy.io.loadmat('orientation-map.mat')
                self.OP_map = data['map']
            except ValueError:
                raise ValueError("Could not find orientation-map.mat!")

        if OD_map == 0:
            self.OD_map = load_OD_map()

        [self.W_EE, self.W_IE, self.W_EI, self.W_II] = generate_connetion_weights( self.N_pairs, self.field_size, self.OP_map, self.kappa_E, self.kappa_I, self.J_EE, self.J_IE, self.J_EI, self.J_II, self.sig_EE, self.sig_IE, self.sig_EI, self.sig_II, self.sig_ori, quiet=True )

        self.sum_field_sizes_E = np.zeros((self.N_pairs, self.N_pairs)) 
        self.sum_field_sizes_I = np.zeros((self.N_pairs, self.N_pairs)) 

    def run_simulation(self, dt, timesteps, c, h, init_cond=np.zeros( (2, 75, 75) ) ):
        r_E = np.zeros((timesteps, self.N_pairs, self.N_pairs))
        r_I = np.copy(r_E)

        # add initial conditions:
        r_E[0,:,:] = init_cond[0]
        r_I[0,:,:] = init_cond[1]

        I_E = np.zeros((timesteps, self.N_pairs, self.N_pairs))
        I_I = np.copy(I_E)
        # rSS_E = np.copy(I_E)
        # rSS_I = np.copy(I_I)

        for t in range(1,timesteps):
            # Input drive from external input and network
            I_E[t,:,:] = c*h + np.sum( np.sum( self.W_EE * r_E[t-1,:,:],1 ), 1 ).reshape(self.N_pairs, self.N_pairs).T - np.sum( np.sum( self.W_EI * r_I[t-1,:,:],1 ), 1 ).reshape(self.N_pairs, self.N_pairs).T 
            I_I[t,:,:] = c*h + np.sum( np.sum( self.W_IE * r_E[t-1,:,:],1 ), 1 ).reshape(self.N_pairs, self.N_pairs).T - np.sum( np.sum( self.W_II * r_I[t-1,:,:],1 ), 1 ).reshape(self.N_pairs, self.N_pairs).T 
            
            # steady state firing rates - power law I/O
            rSS_E = np.multiply(self.k, np.power(np.fmax(0,I_E[t,:,:]), self.n_E))
            rSS_I = np.multiply(self.k, np.power(np.fmax(0,I_I[t,:,:]), self.n_I))

            # set negative steady state rates to zero
            rSS_E[rSS_E < 0] = 0
            rSS_I[rSS_I < 0] = 0

            # instantaneous firing rates approaching steady state
            r_E[t,:,:] = r_E[t-1,:,:] + dt*(np.divide(-r_E[t-1,:,:]+rSS_E, self.tau_E))
            r_I[t,:,:] = r_I[t-1,:,:] + dt*(np.divide(-r_I[t-1,:,:]+rSS_I, self.tau_I))
            
        return [r_E, r_I, I_E, I_I]

    # function to reproduce figure 6B of Rubin et al, 2015, showing the SSN 
    # transition from external to network drive with dominant inhibition
    def plot_network_contrast_response(self, r_units=np.floor( 75*np.random.rand(25,2) ), c_range=np.linspace(3, 50, 12) ):
        pass
    
def diff(x,y):
    return np.abs( np.mod( x - y + 90, 180) - 90 )

def G(x,y,sigma):
    return np.exp(-1*diff(x,y)**2/(2*sigma**2))

def G2D(x_range, y_range, mean, sigma):
    x0 = mean[0]
    y0 = mean[1]
    return np.exp( -1*( ( x_range-x0)**2 + (y_range-y0)**2) / (2*sigma**2) )

def mean_connections(W_ab):
    total = 0.
    for i in range(W_ab.shape[0]):
        sub_mat = W_ab[i,:,:]
        total = total + sub_mat[sub_mat != 0].size
    return total / W_ab.shape[0]

def stimulus_mask(x,length,sig_RF):
    return (1.+np.exp(-(x + length/2.)/sig_RF) )**-1. * (1. - (1.+np.exp(-(x - length/2.)/sig_RF))**-1. )

# generate external drive for an oriented grating stimulus (circular or full frame)
# ori- orientation (degrees)
# size - diameter (degrees)
# centre - position in field of centre of stimulus (degrees, fsize/2 being the centre of the stimulus)
# ocularity - a scaling factor for which eye the stimulus is presented to (1 = contralateral, 0 = ipsilateral) 
# sig_RF - sigma for the stimulus mask
# sig_FF - sigma for full field
# fsize - size of field (degrees, square field)
# full_frame - bool for using the full frame instead of a mask
def generate_ext_stimulus(ori, size, centre, OP_map, OD_map, ocularity, sig_RF=16./75, sig_FF = 32., fsize=16., full_frame=False):
    if ocularity != 0 and ocularity != 1:
        raise ValueError('Ocularity must be either 0 (ipsilateral) or 1 (contralateral).')
    
    if centre[0] > fsize or centre[1] > fsize:
        raise ValueError('Centre of stimulus is off the grid of neurons!')
    
    G_FF = G(ori, OP_map, sig_FF)
    N_pairs = OP_map.shape[0]
    v_range = np.linspace(0, fsize, N_pairs, False)
    
    xv, yv = np.meshgrid( v_range, v_range )
    
    if full_frame==True:
        h = G_FF
    else:
        x_distance = np.abs(xv - centre[0])
        y_distance = np.abs(yv - centre[1])
        dist = np.sqrt(x_distance**2 + y_distance**2)
        mask = stimulus_mask(dist, size, sig_RF)
        h = np.multiply( mask, G_FF )
    
    if ocularity == 1:
        h = h * OD_map
    else:
        h = h * np.abs(OD_map-1)
    
    return h

def generate_mono_stimulus(ori, size, centre, OP_map, sig_RF=16./75, sig_FF=32., fsize=16., full_frame=False):
    if centre[0] > fsize or centre[1] > fsize:
        raise ValueError('Centre of stimulus is off the grid of neurons!')
        
    G_FF = G(ori, OP_map, sig_FF)
    N_pairs = OP_map.shape[0]
    v_range = np.linspace(0, fsize, N_pairs, False)
    
    xv, yv = np.meshgrid( v_range, v_range )
    
    if full_frame==True:
        h = G_FF
    else:
        x_distance = np.abs(xv - centre[0])
        y_distance = np.abs(yv - centre[1])
        dist = np.sqrt(x_distance**2 + y_distance**2)
        mask = stimulus_mask(dist, size, sig_RF)
        h = np.multiply( mask, G_FF )
    
    return h

# generate external drive for an annular stimulus for surround suppression experiments
# orientation - of the stimulus (degrees)
# inner_d - inner diameter of the stimulus (degrees)
# outer_d - inner diameter of the stimulus (degrees
# ocularity - a scaling factor for which eye the stimulus is presented to (1 = contralateral, 0 = ipsilateral)
# mono - boolean for generating a monocular stimulus
# centre - about which the ring is placed
def generate_ring_stimulus(orientation, inner_d, outer_d, centre, ocularity, OP_map, OD_map=0, mono=False, sig_RF=16./75, sig_FF = 32., fsize=16.):
    if centre[0] > fsize or centre[1] > fsize:
        raise ValueError('Centre of stimulus is off the grid of neurons!')
        
    if ocularity != 0 and ocularity != 1 and mono==False:
        raise ValueError('Ocularity must be either 0 (ipsilateral) or 1 (contralateral).')
    
    if inner_d >= outer_d:
        raise ValueError('Inner diameter must be less than the outer diameter (duh).')
    
    G_FF = G(orientation, OP_map, sig_FF)
    N_pairs = OP_map.shape[0]
    v_range = np.linspace(0, fsize, N_pairs, False)
    
    xv, yv = np.meshgrid( v_range, v_range )
    x_distance = np.abs(xv - centre[0])
    y_distance = np.abs(yv - centre[1])
    dist = np.sqrt(x_distance**2 + y_distance**2)
    
    ring_mask = stimulus_mask(dist, outer_d, sig_RF) - stimulus_mask(dist, inner_d, sig_RF)
    
    if mono == False:
        if ocularity == 1:
            h = ring_mask * G_FF * OD_map
        else:
            h = ring_mask * G_FF * np.abs(OD_map-1)
    else:
        h = ring_mask * G_FF
    
    return h

# randomly generate connection weights for all the units in a square grid
def generate_connetion_weights( N_pairs, field_size, OP_map, kappa_E, kappa_I, J_EE, J_IE, J_EI, J_II, sig_EE, sig_IE, sig_EI, sig_II, sig_ori , quiet=False):
	xy_range = np.linspace(0, field_size, N_pairs, False)

	xv, yv = np.meshgrid(xy_range, xy_range) # x and y grid values (degrees)
	G_EE = np.zeros((N_pairs**2, N_pairs, N_pairs))
	G_IE = np.copy(G_EE)

	# may not need these
	G_EI = np.copy(G_EE)
	G_II = np.copy(G_EE)

	G_ori = np.copy(G_EE)
	G_OD = np.copy(G_EE)

	pW_EE = np.copy(G_EE)
	pW_IE = np.copy(G_EE)
	pW_EI = np.copy(G_EE)
	pW_II = np.copy(G_EE)

	rnd_EE = np.copy(G_EE)
	rnd_IE = np.copy(G_EE)
	rnd_EI = np.copy(G_EE)
	rnd_II = np.copy(G_EE)
	np.random.seed(1)

	# iterate through each E/I pair:
	for i in range(N_pairs):
	    for j in range(N_pairs):
	        G_EE[N_pairs*i+j, :, :] = G2D( xv, yv, (xv[0,i] , yv[j,0]), sig_EE)
	        G_IE[N_pairs*i+j, :, :] = G2D( xv, yv, (xv[0,i] , yv[j,0]), sig_IE)
	        G_EI[N_pairs*i+j, :, :] = G2D( xv, yv, (xv[0,i] , yv[j,0]), sig_EI)
	        G_II[N_pairs*i+j, :, :] = G2D( xv, yv, (xv[0,i] , yv[j,0]), sig_II)
	        
	        G_ori[N_pairs*i+j,:,:] = G(OP_map[j,i], OP_map, sig_ori)
	        
	        # Does ocular dominance affect connectivity? 
	        # Not according to Lowel and Singer, 1992, pg. 210-11:
	        # "Analyses... provided no evidence for eye-specific selectivity of tangential connections"
	        # Leaving this commented for future experiments though:
	#         G_OD[N_pairs*i+j,:,:] = G(OD_map[j,i], OD_map, sig_OD)
	        
	        rnd_EE[N_pairs*i+j, :, :] = np.random.rand(N_pairs, N_pairs)
	        rnd_IE[N_pairs*i+j, :, :] = np.random.rand(N_pairs, N_pairs)
	        rnd_EI[N_pairs*i+j, :, :] = np.random.rand(N_pairs, N_pairs)
	        rnd_II[N_pairs*i+j, :, :] = np.random.rand(N_pairs, N_pairs)

	for i in range(N_pairs**2):
	    pW_EE[i,:,:] = kappa_E * np.multiply(G_EE[i,:,:], G_ori[i,:,:])
	    pW_IE[i,:,:] = kappa_E * np.multiply(G_IE[i,:,:], G_ori[i,:,:])
	    pW_EI[i,:,:] = kappa_I * np.multiply(G_EI[i,:,:], G_ori[i,:,:])
	    pW_II[i,:,:] = kappa_I * np.multiply(G_II[i,:,:], G_ori[i,:,:])
	      # for OD connectivity experiements:
	#     pW_EE[i,:,:] = kappa_E * G_EE[i,:,:] * G_ori[i,:,:] * G_OD[i,:,:]
	#     pW_IE[i,:,:] = kappa_E * G_IE[i,:,:] * G_ori[i,:,:] * G_OD[i,:,:]
	#     pW_EI[i,:,:] = kappa_I * G_EI[i,:,:] * G_ori[i,:,:] * G_OD[i,:,:]
	#     pW_II[i,:,:] = kappa_I * G_II[i,:,:] * G_ori[i,:,:] * G_OD[i,:,:]
	    
	# find zero-weighted connections:
	W_EE = np.ones((N_pairs**2, N_pairs, N_pairs))
	W_IE = np.copy(W_EE)
	W_EI = np.copy(W_EE)
	W_II = np.copy(W_EE)

	W_EE[pW_EE<rnd_EE] = 0
	W_IE[pW_IE<rnd_IE] = 0
	W_EI[pW_EI<rnd_EI] = 0
	W_II[pW_II<rnd_II] = 0

	u_EE = mean_connections(W_EE)
	u_IE = mean_connections(W_IE)
	u_EI = mean_connections(W_EI)
	u_II = mean_connections(W_II)
	if quiet==False:
		print "Mean no. of connections:\nu_EE: %d\t u_IE: %d\t u_EI: %d\t u_II: %d" % (u_EE, u_IE, u_EI, u_II)

	# For non-zero connections, determine the weight 
	W_EE[W_EE != 0] = np.random.normal(J_EE, 0.25*J_EE, W_EE[W_EE!=0].size)
	W_IE[W_IE != 0] = np.random.normal(J_IE, 0.25*J_IE, W_IE[W_IE!=0].size)
	W_EI[W_EI != 0] = np.random.normal(J_EI, 0.25*J_EI, W_EI[W_EI!=0].size)
	W_II[W_II != 0] = np.random.normal(J_II, 0.25*J_II, W_II[W_II!=0].size)

	# Set negative weights to zero:
	W_EE[W_EE < 0] = 0
	W_IE[W_IE < 0] = 0
	W_EI[W_EI < 0] = 0
	W_II[W_II < 0] = 0

	# "Weights of a given type 'b' onto each unit 
	# are then scaled so that all units of a given type 'a' receive the same 
	# total type b synaptic weight, equal to Jab times the mean number of 
	# connections received under probabilistic function
	for i in range(N_pairs**2):
	    if np.all(W_EE[i,:,:] == np.zeros((N_pairs, N_pairs))) == False:
	        W_EE[i,:,:] = W_EE[i,:,:]*J_EE*u_EE/np.sum(W_EE[i,:,:])
	    
	    if np.all(W_IE[i,:,:] == np.zeros((N_pairs, N_pairs))) == False:
	        W_IE[i,:,:] = W_IE[i,:,:]*J_IE*u_IE/np.sum(W_IE[i,:,:])

	    if np.all(W_EI[i,:,:] == np.zeros((N_pairs, N_pairs))) == False:
	        W_EI[i,:,:] = W_EI[i,:,:]*J_EI*u_EI/np.sum(W_EI[i,:,:])

	    if np.all(W_II[i,:,:] == np.zeros((N_pairs, N_pairs))) == False:
	        W_II[i,:,:] = W_II[i,:,:]*J_II*u_II/np.sum(W_II[i,:,:])

    # From S.1.3.2: for strongest nonlinear behaviour, omega_E < 0 and omega_E < omega_I
	# where omega_E = sum(W_II) - sum(W_EI), omega_I = sum(W_IE) - sum(W_EE)
	# Verify here:
	if quiet==False:
		omega_E = np.sum(W_II) - np.sum(W_EI)
		omega_I = np.sum(W_IE) - np.sum(W_EE)

		print 'Omega_E: ', omega_E
		print 'Omega_I: ', omega_I

		if omega_E < 0 and omega_I > omega_E:
		    print "System should show strong nonlinear behaviour!"
		else:
		    print "System may not show strong nonlinear behaviour."

	return [W_EE, W_IE, W_EI, W_II]


def run_simulation( dt, timesteps, c, h, k, n_E, n_I, tau_E, tau_I, W_EE, W_EI, W_IE, W_II, init_cond=[np.zeros((75, 75)),np.zeros((75, 75))]):
    N_pairs = W_EE.shape[1]
    r_E = np.zeros((timesteps, N_pairs, N_pairs))
    r_I = np.copy(r_E)

    # add initial conditions:
    r_E[0,:,:] = init_cond[0]
    r_I[0,:,:] = init_cond[1]

    I_E = np.zeros((timesteps, N_pairs, N_pairs))
    I_I = np.copy(I_E)
    # rSS_E = np.copy(I_E)
    # rSS_I = np.copy(I_I)

    for t in range(1,timesteps):
        # Input drive from external input and network
        I_E[t,:,:] = c*h + np.sum( np.sum( W_EE * r_E[t-1,:,:],1 ), 1 ).reshape(N_pairs, N_pairs).T - np.sum( np.sum( W_EI * r_I[t-1,:,:],1 ), 1 ).reshape(N_pairs, N_pairs).T 
        I_I[t,:,:] = c*h + np.sum( np.sum( W_IE * r_E[t-1,:,:],1 ), 1 ).reshape(N_pairs, N_pairs).T - np.sum( np.sum( W_II * r_I[t-1,:,:],1 ), 1 ).reshape(N_pairs, N_pairs).T 
        
        # steady state firing rates - power law I/O
        rSS_E = np.multiply(k, np.power(np.fmax(0,I_E[t,:,:]), n_E))
        rSS_I = np.multiply(k, np.power(np.fmax(0,I_I[t,:,:]), n_I))

        # set negative steady state rates to zero
        rSS_E[rSS_E < 0] = 0
        rSS_I[rSS_I < 0] = 0

        # instantaneous firing rates approaching steady state
        r_E[t,:,:] = r_E[t-1,:,:] + dt*(np.divide(-r_E[t-1,:,:]+rSS_E, tau_E))
        r_I[t,:,:] = r_I[t-1,:,:] + dt*(np.divide(-r_I[t-1,:,:]+rSS_I, tau_I))
        
    return [r_E, r_I, I_E, I_I]

# finds the summation field size of a given unit
def find_sum_field_size(cell_ind, OP_map, W_EE, W_EI, W_IE, W_II, OD_map=1, n_sizes=10, max_size=5):
	stim_sizes = np.linspace(0.5, max_size, n_sizes)
	xi = cell_ind[0]
	yi = cell_ind[1]
	ori = OP_map[yi,xi]

	k   = np.random.normal(0.012, 0.05*0.012, (N_pairs, N_pairs))
	n_E = np.random.normal(2.0, 0.05*2.0, (N_pairs, N_pairs)) 
	n_I = np.random.normal(2.2, 0.05*2.2, (N_pairs, N_pairs))
	tau_E = np.random.normal(0.02, 0.05*0.02, (N_pairs, N_pairs))
	tau_I = np.random.normal(0.01, 0.05*0.01, (N_pairs, N_pairs))
	dt = 0.005
	timesteps = 100

