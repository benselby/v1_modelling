#!/usr/bin/env python

###
# A mini library containing the functions typically used when running 
# simulations using the supralinear stabilized network (Rubin et al., 2015).
# 
# Ben Selby, September 2015

# import os
# try:
    # print "Trying to access GPU for Theano..."
    # os.environ['THEANO_FLAGS'] = 'device=gpu, floatX=float32'
# except Error:
    # print "No GPU detected, moving on!"
    # os.environ['THEANO_FLAGS'] = 'floatX=float32'

import numpy as np
import scipy.io
import matplotlib.image as mpimg

import theano
import theano.tensor as T

import tensorflow as tf

class SSNetwork:

    # Default constructor - use the Rubin et al. parameters to produce a SSN layer:
    def __init__(self, sig_EE=8, sig_IE=12, sig_EI=4, sig_II=4,
                       J_EE=0.1, J_IE=0.38, J_EI=0.089, J_II=0.096, 
                       ori_map=0, ocd_map=None, od_bias=0., 
                       N_pairs=75, field_size=16., subpop=True, subpop_size=25):

        self.N_pairs = N_pairs # no. of E/I pairs to a side of a grid

        self.field_size = field_size # size of field to a side (degrees)

        print "Generating an SSN with grid size %d over %2.1f degrees." % (self.N_pairs, self.field_size)

        self.dx = field_size / N_pairs

        self.sig_FF = 32.
        self.sig_RF = self.dx

        self.k   = np.random.normal(0.012, 0.05*0.012, (N_pairs, N_pairs))
        self.n_E = np.random.normal(2.0, 0.05*2.0, (N_pairs, N_pairs)) 
        self.n_I = np.random.normal(2.2, 0.05*2.2, (N_pairs, N_pairs))

        # Generate subunit populations for the target supralinear responses:
        if subpop:
            self.subunit_pops_E = []
            self.subunit_pops_I = []
            self.subpop_size = subpop_size

            self.subpop_weights_E = np.zeros((self.N_pairs**2, self.subpop_size))
            self.subpop_weights_I = np.copy(self.subpop_weights_E)

            self.subunit_T_fxns_E = []
            self.subunit_T_fxns_I = []

            for i in range(self.N_pairs**2):
                
                xi = np.floor(i/self.N_pairs)
                yi = i%self.N_pairs
                self.subunit_pops_E.append(SublinearPopulation(self.k[yi,xi],self.n_E[yi,xi],pop_size=self.subpop_size))
                self.subunit_pops_I.append(SublinearPopulation(self.k[yi,xi],self.n_I[yi,xi],pop_size=self.subpop_size))

                self.subpop_weights_E[i,:] = np.squeeze(self.subunit_pops_E[i].weights)
                self.subpop_weights_I[i,:] = np.squeeze(self.subunit_pops_I[i].weights)

                # self.subunit_T_fxns_E.append( self.subunit_pops_E[i].t_fxn )
                # self.subunit_T_fxns_I.append( self.subunit_pops_I[i].t_fxn ) 

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

        self.OD_bias_weight = od_bias
        if od_bias == 0:
            OD_dependent = False
        else:
            OD_dependent = True

        self.sig_EE = sig_EE*self.dx
        self.sig_IE = sig_IE*self.dx
        self.sig_EI = sig_EI*self.dx
        self.sig_II = sig_II*self.dx
        self.sig_ori = 45.

        self.OP_map = ori_map
        if np.all(self.OP_map == 0):
            try:
                # load OP map from Bryan's extracted Kaschube map
                data = scipy.io.loadmat('orientation-map.mat')
                self.OP_map = data['map']
            except ValueError:
                raise ValueError("Could not find orientation-map.mat!")
        
        if self.N_pairs!=self.OP_map.shape[0]:
            self.OP_map = self.OP_map[:self.N_pairs,:self.N_pairs]

        self.OD_map = ocd_map
        if self.OD_map is None:
            self.OD_map = np.zeros((self.N_pairs, self.N_pairs))
            print "Instantiating a SSN without ocular domiance map."

        [self.W_EE, self.W_IE, self.W_EI, self.W_II] = generate_connection_weights( self.N_pairs, self.field_size, self.OP_map, self.kappa_E, self.kappa_I, self.J_EE, self.J_IE, self.J_EI, self.J_II, self.sig_EE, self.sig_IE, self.sig_EI, self.sig_II, self.sig_ori, quiet=True, OD_map=self.OD_map, OD_dependent=OD_dependent,OD_bias_weight=self.OD_bias_weight )

        # initial firing rates are all zero
        self.r_E = theano.shared(np.zeros((self.N_pairs,self.N_pairs), dtype='float32'))
        self.r_I = theano.shared(np.zeros((self.N_pairs,self.N_pairs), dtype='float32'))

        self.increment_simulation = generate_theano_simulation(self.r_E, self.r_I, self.N_pairs)

        self.get_next_inputs = generate_subpop_input(self.r_E, self.r_I, self.N_pairs)
        self.increment_subpop_simulation = generate_subpop_firing(self.r_E, self.r_I, self.N_pairs)


    def run_simulation(self, c0, h0, timesteps=100, dt0=0.005):
        self.r_E.set_value(np.zeros((self.N_pairs,self.N_pairs),dtype='float32'))
        self.r_I.set_value(np.zeros((self.N_pairs,self.N_pairs),dtype='float32'))

        rE_out = np.zeros((timesteps, self.N_pairs, self.N_pairs))
        rI_out = np.copy(rE_out)
        rss_E_out = np.copy(rE_out)
        rss_I_out = np.copy(rI_out)
        
        for t in range(timesteps):
            [rss_E, rss_I] = self.increment_simulation(dt0,c0,h0,self.W_EE,self.W_EI,self.W_IE,self.W_II,self.n_E,self.n_I,self.k,self.tau_E,self.tau_I)
            rE_out[t] = self.r_E.get_value()
            rI_out[t] = self.r_I.get_value()
            rss_E_out[t] = rss_E
            rss_I_out[t] = rss_I
        # resp_E = self.r_E.get_value()
        # resp_I = self.r_I.get_value()

        self.r_E.set_value(np.zeros((self.N_pairs,self.N_pairs),dtype='float32'))
        self.r_I.set_value(np.zeros((self.N_pairs,self.N_pairs),dtype='float32'))
        # return resp_E, resp_I
        return rE_out, rI_out, rss_E_out, rss_I_out

    def run_tf_simulation(self, c_in, h_in, timesteps=100, dt=0.005):
        r_e = tf.Variable( tf.zeros([self.N_pairs, self.N_pairs]) )
        r_i = tf.Variable( tf.zeros([self.N_pairs, self.N_pairs]) )
        
        W_EE = tf.placeholder(tf.float32)
        W_EI = tf.placeholder(tf.float32)
        W_IE = tf.placeholder(tf.float32)
        W_II = tf.placeholder(tf.float32)
        k = tf.placeholder(tf.float32)
        n_E = tf.placeholder(tf.float32)
        n_I = tf.placeholder(tf.float32) 
        tau_E = tf.placeholder(tf.float32)
        tau_I = tf.placeholder(tf.float32)
        
        c0 = tf.constant(c_in)
        h0 = tf.constant(h_in)
                
        # Compile functions:
        I_E = c0*h0 + tf.transpose(tf.reshape(tf.reduce_sum(W_EE * r_e, [1,2]), [75,75])) \
            - tf.transpose(tf.reshape(tf.reduce_sum(W_EI * r_i, [1,2]), [75,75]))
        I_I = c0*h0 + tf.transpose(tf.reshape(tf.reduce_sum(W_IE * r_e, [1,2]), [75,75])) \
            - tf.transpose(tf.reshape(tf.reduce_sum(W_II * r_i, [1,2]), [75,75]))

        I_thresh_E = tf.maximum(0., I_E)
        I_thresh_I = tf.maximum(0., I_I)

        r_SS_E = k * tf.pow(I_thresh_E, n_E)
        r_SS_I = k * tf.pow(I_thresh_I, n_I)

        rE_out = r_e + dt*(-r_e+r_SS_E)/tau_E
        rI_out = r_i + dt*(-r_i+r_SS_I)/tau_I
        
        update_rE = tf.assign(r_e, rE_out)
        update_rI = tf.assign(r_i, rI_out)
        
        init = tf.initialize_all_variables()
        
        rE = 0
        rI = 0
        
        fd = {W_EE:self.W_EE.astype(np.float32), 
                  W_EI:self.W_EI.astype(np.float32), 
                  W_IE:self.W_IE.astype(np.float32), 
                  W_II:self.W_II.astype(np.float32),
                  k:self.k.astype(np.float32),
                  n_E:self.n_E.astype(np.float32),
                  n_I:self.n_I.astype(np.float32),
                  tau_E:self.tau_E.astype(np.float32),
                  tau_I:self.tau_I.astype(np.float32)}
        
        with tf.Session() as sess:
            sess.run(init, feed_dict=fd)
            for t in range(timesteps):
                # run the simulation
                sess.run([update_rE, update_rI], feed_dict=fd)
            # fetch the rates
            rE = sess.run([r_e], feed_dict=fd)
            rI = sess.run([r_i], feed_dict=fd)
            
        return rE, rI
        
    def run_subpop_simulation(self, c_in, h_in, timesteps=100, dt=0.005):
        r_e = tf.Variable( tf.zeros([self.N_pairs, self.N_pairs]) )
        r_i = tf.Variable( tf.zeros([self.N_pairs, self.N_pairs]) )
        
        W_EE = tf.placeholder(tf.float32)
        W_EI = tf.placeholder(tf.float32)
        W_IE = tf.placeholder(tf.float32)
        W_II = tf.placeholder(tf.float32)
        k = tf.placeholder(tf.float32)
        n_E = tf.placeholder(tf.float32)
        n_I = tf.placeholder(tf.float32) 
        tau_E = tf.placeholder(tf.float32)
        tau_I = tf.placeholder(tf.float32)
        
        subpop_W_E = tf.placeholder(tf.float32)
        subpop_W_I = tf.placeholder(tf.float32)
        
        c0 = tf.constant(c_in)
        h0 = tf.constant(h_in)
                
        # Compile functions:
        I_E = c0*h0 + tf.transpose(tf.reshape(tf.reduce_sum(W_EE * r_e, [1,2]), [75,75])) \
            - tf.transpose(tf.reshape(tf.reduce_sum(W_EI * r_i, [1,2]), [75,75]))
        I_I = c0*h0 + tf.transpose(tf.reshape(tf.reduce_sum(W_IE * r_e, [1,2]), [75,75])) \
            - tf.transpose(tf.reshape(tf.reduce_sum(W_II * r_i, [1,2]), [75,75]))

        I_thresh_E = tf.maximum(0.,I_E)
        I_thresh_I = tf.maximum(0.,I_I)
        
        r_SS_E, r_SS_I = self.get_subpop_responses(I_E, I_I)
        
        rE_out = r_e + dt*(-r_e+r_SS_E)/tau_E
        rI_out = r_i + dt*(-r_i+r_SS_I)/tau_I
        
        update_rE = tf.assign(r_e, rE_out)
        update_rI = tf.assign(r_i, rI_out)
        
        init = tf.initialize_all_variables()
        
        rE = 0
        rI = 0
        
        fd = {W_EE:self.W_EE.astype(np.float32), 
              W_EI:self.W_EI.astype(np.float32), 
              W_IE:self.W_IE.astype(np.float32), 
              W_II:self.W_II.astype(np.float32),
              k:self.k.astype(np.float32),
              n_E:self.n_E.astype(np.float32),
              n_I:self.n_I.astype(np.float32),
              tau_E:self.tau_E.astype(np.float32),
              tau_I:self.tau_I.astype(np.float32),
              subpop_W_E:self.subpop_weights_E,
              subpop_W_I:self.subpop_weights_I}
        
        with tf.Session() as sess:
            sess.run(init, feed_dict=fd)
            for t in range(timesteps):
                # run the simulation
                sess.run([update_rE, update_rI], feed_dict=fd)
            rE = sess.run([rE_out], feed_dict=fd)
            rI = sess.run([rI_out], feed_dict=fd)
            
        return rE, rI
        
    def get_subpop_responses(self, I_E, I_I):
        pop_resps_E = [[]]
        pop_resps_I = [[]]
        
        for i in range(self.N_pairs**2):
            xi = np.floor(i/self.N_pairs)
            yi = i%self.N_pairs         
            tf.concat( 0, [pop_resps_E, self.subunit_pops_E[i].get_tf_responses( I_E[yi,xi] )] )
            tf.concat( 0, [pop_resps_I, self.subunit_pops_I[i].get_tf_responses( I_I[yi,xi] )] )
            
        r_SS_E = tf.reshape(tf.reduce_sum(pop_resps_E*self.subpop_weights_E,1), [self.N_pairs, self.N_pairs])
        r_SS_I = tf.reshape(tf.reduce_sum(pop_resps_I*self.subpop_weights_I,1), [self.N_pairs, self.N_pairs])
        
        return r_SS_E, r_SS_I

def generate_theano_simulation(r_E_shared, r_I_shared, n_pairs):
    dt = T.scalar('dt', dtype='float32')
    c = T.scalar("c", dtype='float32')
    h = T.matrix("h", dtype='float32')
    n_E = T.matrix("n_E", dtype='float32')
    n_I = T.matrix("n_I", dtype='float32')
    W_EE = T.tensor3("W_EE", dtype='float32')
    W_EI = T.tensor3("W_EI", dtype='float32')
    W_IE = T.tensor3("W_IE", dtype='float32')
    W_II = T.tensor3("W_II", dtype='float32')

    k = T.matrix("k", dtype='float32')
    tau_E = T.matrix("tau_E", dtype='float32')
    tau_I = T.matrix("tau_I", dtype='float32')

    I_E = T.matrix('I_E', dtype='float32')
    I_I = T.matrix('I_I', dtype='float32')

    I_thresh_E = T.matrix('I_thresh_E', dtype='float32')
    I_thresh_I = T.matrix('I_thresh_I', dtype='float32')

    r_SS_E = T.matrix('r_SS_E', dtype='float32')
    r_SS_I = T.matrix('r_SS_I', dtype='float32')

    r_e = T.matrix("r_e", dtype='float32')
    r_i = T.matrix("r_i", dtype='float32')

    # Compile functions:
    I_E = c*h + T.sum(T.sum(W_EE*r_e,1),1).reshape((75,75)).T - T.sum(T.sum(W_EI*r_i,1),1).reshape((75,75)).T
    I_I = c*h + T.sum(T.sum(W_IE*r_e,1),1).reshape((75,75)).T - T.sum(T.sum(W_II*r_i,1),1).reshape((75,75)).T

    I_thresh_E = T.switch(T.lt(I_E,0), 0, I_E)
    I_thresh_I = T.switch(T.lt(I_I,0), 0, I_I)

    r_SS_E = k*T.pow(I_thresh_E, n_E)
    r_SS_I = k*T.pow(I_thresh_I, n_I)

    euler_E = r_e + dt*(-r_e+r_SS_E)/tau_E
    euler_I = r_i + dt*(-r_i+r_SS_I)/tau_I

    euler = theano.function(inputs=[dt,c,h,W_EE,W_EI,W_IE,W_II,n_E,n_I,k,tau_E,tau_I], 
                                outputs=[r_SS_E, r_SS_I],
                                givens={r_e:r_E_shared, r_i:r_I_shared},
                                updates=[(r_E_shared,euler_E), (r_I_shared,euler_I)],
                                allow_input_downcast=True)
    return euler

    
def generate_subpop_input(r_E, r_I, n_pairs):
    
    c = T.scalar("c", dtype='float32')
    h = T.matrix("h", dtype='float32')
    W_EE = T.tensor3("W_EE", dtype='float32')
    W_EI = T.tensor3("W_EI", dtype='float32')
    W_IE = T.tensor3("W_IE", dtype='float32')
    W_II = T.tensor3("W_II", dtype='float32')

    r_e = T.matrix("r_e", dtype='float32')
    r_i = T.matrix("r_i", dtype='float32')

    I_E = T.matrix('I_E', dtype='float32')
    I_I = T.matrix('I_I', dtype='float32')

    I_thresh_E = T.matrix('I_thresh_E', dtype='float32')
    I_thresh_I = T.matrix('I_thresh_I', dtype='float32')

    # Compile functions:
    I_E = c*h + T.sum(T.sum(W_EE*r_e,1),1).reshape((n_pairs, n_pairs)).T - T.sum(T.sum(W_EI*r_i,1),1).reshape((n_pairs, n_pairs)).T
    I_I = c*h + T.sum(T.sum(W_IE*r_e,1),1).reshape((n_pairs, n_pairs)).T - T.sum(T.sum(W_II*r_i,1),1).reshape((n_pairs, n_pairs)).T

    I_thresh_E = T.switch(T.lt(I_E,0), 0, I_E)
    I_thresh_I = T.switch(T.lt(I_I,0), 0, I_I)

    inputs = theano.function(inputs=[c,h,W_EE,W_EI,W_IE,W_II],
                                outputs=[I_thresh_E, I_thresh_I],
                                givens={r_e:r_E, r_i:r_I},
                                allow_input_downcast=True)
    return inputs

    
def generate_subpop_firing(r_E, r_I, n_pairs):
    dt = T.scalar('dt', dtype='float32')

    pop_resps_E = T.matrix("sE", dtype='float32')
    pop_resps_I = T.matrix("sI", dtype='float32')
    weights_E = T.matrix("wE", dtype='float32')
    weights_I = T.matrix("wI", dtype='float32')

    tau_E = T.matrix("tau_E", dtype='float32')
    tau_I = T.matrix("tau_I", dtype='float32')

    r_e = T.matrix("r_e", dtype='float32')
    r_i = T.matrix("r_i", dtype='float32')

    r_SS_E = T.matrix('r_SS_E', dtype='float32')
    r_SS_I = T.matrix('r_SS_I', dtype='float32')

    r_SS_E = T.sum(pop_resps_E*weights_E, 1).reshape((n_pairs, n_pairs))
    r_SS_I = T.sum(pop_resps_I*weights_I, 1).reshape((n_pairs, n_pairs))

    euler_E = r_e + dt*(-r_e+r_SS_E)/tau_E
    euler_I = r_i + dt*(-r_i+r_SS_I)/tau_I

    euler = theano.function(inputs=[dt,pop_resps_E,pop_resps_I,weights_E, weights_I,tau_E,tau_I], 
                            outputs=[r_E, r_I],
                            givens={r_e:r_E, r_i:r_I},
                            updates=[(r_E,euler_E), (r_I,euler_I)],
                            allow_input_downcast=True)
    return euler

def select_random_units(n_units, N_pairs=75):
    return np.floor(N_pairs*np.random.rand(n_units,2))
    
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
def generate_connection_weights( N_pairs, field_size, OP_map, kappa_E, kappa_I, J_EE, J_IE, J_EI, J_II, sig_EE, sig_IE, sig_EI, sig_II, sig_ori , quiet=False, OD_map=None, OD_dependent=False, OD_bias_weight=0.):
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
            
            rnd_EE[N_pairs*i+j, :, :] = np.random.rand(N_pairs, N_pairs)
            rnd_IE[N_pairs*i+j, :, :] = np.random.rand(N_pairs, N_pairs)
            rnd_EI[N_pairs*i+j, :, :] = np.random.rand(N_pairs, N_pairs)
            rnd_II[N_pairs*i+j, :, :] = np.random.rand(N_pairs, N_pairs)

    for i in range(N_pairs**2):
        pW_EE[i,:,:] = kappa_E * np.multiply(G_EE[i,:,:], G_ori[i,:,:])
        pW_IE[i,:,:] = kappa_E * np.multiply(G_IE[i,:,:], G_ori[i,:,:])
        pW_EI[i,:,:] = kappa_I * np.multiply(G_EI[i,:,:], G_ori[i,:,:])
        pW_II[i,:,:] = kappa_I * np.multiply(G_II[i,:,:], G_ori[i,:,:])
        
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

    od_round = np.round(OD_map)
    if OD_dependent==True:
        print "Generating connection weights biased for ocular dominance..."
    # "Weights of a given type 'b' onto each unit 
    # are then scaled so that all units of a given type 'a' receive the same 
    # total type b synaptic weight, equal to Jab times the mean number of 
    # connections received under probabilistic function
    for i in range(N_pairs**2):
        OD_bias = np.ones((N_pairs,N_pairs))
        if OD_dependent==True:
            if OD_map is None:
                raise ValueError("Cannot generate ocular dominance-dependent connection weights because no map is specified.")
            xi = np.floor(i/N_pairs)
            yi = i - N_pairs*xi
            od_pref = od_round[yi,xi]
            if od_pref == 1:
                OD_bias = OD_bias - OD_bias_weight*(np.abs(OD_map-1))
            else:
                OD_bias = OD_bias - OD_bias_weight*OD_map

        if np.all(W_EE[i,:,:] == np.zeros((N_pairs, N_pairs))) == False:
            W_EE[i,:,:] = W_EE[i,:,:]*J_EE*u_EE/np.sum(W_EE[i,:,:])*OD_bias
        
        if np.all(W_IE[i,:,:] == np.zeros((N_pairs, N_pairs))) == False:
            W_IE[i,:,:] = W_IE[i,:,:]*J_IE*u_IE/np.sum(W_IE[i,:,:])*OD_bias

        if np.all(W_EI[i,:,:] == np.zeros((N_pairs, N_pairs))) == False:
            W_EI[i,:,:] = W_EI[i,:,:]*J_EI*u_EI/np.sum(W_EI[i,:,:])*OD_bias

        if np.all(W_II[i,:,:] == np.zeros((N_pairs, N_pairs))) == False:
            W_II[i,:,:] = W_II[i,:,:]*J_II*u_II/np.sum(W_II[i,:,:])*OD_bias

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


class SublinearPopulation:
    
    def __init__(self, k, n, pop_fxns=None, pop_size=25):
        self.contrast_range = np.linspace(1, 100, 1000)
        if pop_fxns:
            self.pop_size = len(pop_fxns)
            self.pop_fxns = pop_fxns
            self.crf_evals = self.get_neuron_responses(self.contrast_range)
        else:
            self.pop_size = pop_size
            self.pop_fxns, self.crf_evals = self.generate_pop_fxns()
        
        self.weights = self.find_weights(k, n)
        
    def generate_pop_fxns(self):
        crf_evals = np.zeros((self.pop_size, len(self.contrast_range)))
        crf_type_dist = np.array([.04, .19, .07, .70])
        crf_type_cdf = np.cumsum(crf_type_dist)
        pop_size = float(self.pop_size)

        pop_fxns = []

        for i in range(self.pop_size):
            if i/pop_size < crf_type_cdf[0]:
                a = 0.25*50*np.random.randn() - 50
                b = 2*np.random.rand() + 0.5
                crf_evals[i,:] = linear_crf(a, b, self.contrast_range)
                pop_fxns.append({'ftype':'linear', 'a':a, 'b':b})
            elif i/pop_size >= crf_type_cdf[0] and i/pop_size < crf_type_cdf[1]:
                a = 0.25*43*np.random.randn() - 43
                b = 0.25*80*np.random.randn() + 80
                crf_evals[i,:] = log_crf(a, b, self.contrast_range)
                pop_fxns.append({'ftype':'log', 'a':a, 'b':b})
            elif i/pop_size >= crf_type_cdf[1] and i/pop_size < crf_type_cdf[2]:
                a = 0.25*7.7*np.random.randn() + 7.7
                b = 0.25*0.63*np.random.randn() + 0.63
                crf_evals[i,:] = power_crf(a, b, self.contrast_range)
                pop_fxns.append({'ftype':'power', 'a':a, 'b':b})
            else:
                # from table 5:
                r_max = 2.7*np.random.randn() + 115.0
                c_50 = 0.9*np.random.randn() + 19.3
                n = 0.1*np.random.randn() + 2.9
                scale = 1.5*np.random.rand()
                crf_evals[i,:] = scale*h_ratio_crf(r_max, c_50, n, self.contrast_range)
                pop_fxns.append({'ftype':'h_ratio', 'r_max':r_max, 'n':n, 'c_50':c_50, 'scale':scale})
        
        return pop_fxns, crf_evals
    
    def find_weights(self, k, n):
        target_crf = k*self.contrast_range**n
        
#         weights = np.linalg.lstsq(self.crf_evals.T,target_crf)[0]
        pinv = np.linalg.pinv( self.crf_evals.T )
        weights = np.dot(pinv, target_crf)
        return weights[:,np.newaxis]
    
    def get_neuron_responses(self, drive):
        neuron_responses = np.zeros((self.pop_size, len(drive)))
        for i in range(self.pop_size):
            if self.pop_fxns[i]['ftype']=='linear':
                neuron_responses[i] = linear_crf(self.pop_fxns[i]['a'],self.pop_fxns[i]['b'],drive)
            elif self.pop_fxns[i]['ftype']=='log':
                neuron_responses[i] = log_crf(self.pop_fxns[i]['a'],self.pop_fxns[i]['b'],drive)
            elif self.pop_fxns[i]['ftype']=='power':
                neuron_responses[i] = power_crf(self.pop_fxns[i]['a'],self.pop_fxns[i]['b'],drive)
            else:
                neuron_responses[i] = self.pop_fxns[i]['scale']*h_ratio_crf(self.pop_fxns[i]['r_max'],self.pop_fxns[i]['c_50'],self.pop_fxns[i]['n'],drive)
    
        return neuron_responses
    
    def get_tf_responses(self, drive):
        neuron_responses = []
        for i in range(self.pop_size):
            if self.pop_fxns[i]['ftype']=='linear':
                neuron_responses.append( linear_crf_tf(self.pop_fxns[i]['a'],self.pop_fxns[i]['b'],drive) )
            elif self.pop_fxns[i]['ftype']=='log':
                neuron_responses.append( log_crf_tf(self.pop_fxns[i]['a'],self.pop_fxns[i]['b'],drive) )
            elif self.pop_fxns[i]['ftype']=='power':
                neuron_responses.append( power_crf_tf(self.pop_fxns[i]['a'],self.pop_fxns[i]['b'],drive) )
            else:
                neuron_responses.append( self.pop_fxns[i]['scale']*h_ratio_crf_tf(self.pop_fxns[i]['r_max'],self.pop_fxns[i]['c_50'],self.pop_fxns[i]['n'],drive) )
    
        return neuron_responses
    
    def get_subunit_response(self, drive):
        neuron_responses = self.get_neuron_responses(drive)
        return neuron_responses.T.dot(self.weights)
   
def linear_crf(a, b, x):
    return np.fmax(0,a + b*x)

def log_crf(a,b,x):
    return np.fmax(0, a + b*np.log10(x))

def power_crf(a,b,x):
    return np.fmax(0, a*x**b)

def h_ratio_crf(r_max, c_50, n, x):
    return np.fmax(0, r_max*(x**n/(x**n + c_50**n)))
    
### TENSORFLOW VERSIONS OF ALL CRF FUNCTIONS ### 
 
def linear_crf_tf(a, b, x):
    return tf.maximum(0.,a + b*x)

def log_crf_tf(a,b,x):
    # TensorFlow has no base-10 log function, so convert from 
    # natural log instead:
    # a = np.float32(a_in)
    # b = np.float32(b_in)
    # x = np.float32(x_in)
    return tf.maximum(0., a + b*tf.log(x)/2.303 )

def power_crf_tf(a,b,x):
    return tf.maximum(0., a*x**b)

def h_ratio_crf_tf(r_max, c_50, n, x):
    return tf.maximum(0., r_max*(x**n/(x**n + c_50**n)))
    

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