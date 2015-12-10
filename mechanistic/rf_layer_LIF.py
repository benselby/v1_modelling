import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import time

def gabor(sig_x, sig_y, theta, k, phi, fsize, sine=False): 
    vals = np.linspace(-np.floor(fsize/2), np.floor(fsize/2), fsize)
    xv,yv= np.meshgrid(vals,vals)
    Xj = xv*np.cos(theta) - yv*np.sin(theta);
    Yj = xv*np.sin(theta) + yv*np.cos(theta);
    if sine == False:
        gabor = (1/(2*np.pi*sig_x*sig_y))*np.exp(-1*Xj**2/(2*sig_x**2) - Yj**2/(2*sig_y**2) )*np.cos(2*np.pi*k*Xj-phi);
    else:
        gabor = (1/(2*np.pi*sig_x*sig_y))*np.exp(-1*Xj**2/(2*sig_x**2) - Yj**2/(2*sig_y**2) )*np.sin(2*np.pi*k*Xj-phi);
    return gabor

# generate a single circular sinusoidal grating with the specified orientation and SF
# theta - grating orientation (radians)
# diameter - total stimulus size (grating and mask) (degrees)
# SF - spatial frequency (cycles / frame)
# pix_deg - number of pixels per degree scale factor
# phi - sinusoid phase shift (radians)
# masked - flag for surrounding the grating in a grey circular mask
# mask_size - grating diameter (degrees)
# norm - flag for normalizing the grating values from zero to one
def generate_static_grating(theta, diameter, SF, pix_deg, contrast=1, phi=0, masked=False, mask_size=0, norm=True, mask_pos=[0,0]):
    fsize = pix_deg * diameter # pixels per size of img
    
    vals = np.linspace(-np.pi, np.pi, fsize)
    xv, yv = np.meshgrid(vals, vals)
    
    xy = xv*np.cos(theta) - yv*np.sin(theta)
    
    mask = np.ones((fsize, fsize))

    
    if masked:
        my,mx = np.ogrid[-fsize/2:fsize/2, -fsize/2:fsize/2] 
        
        xs = pix_deg*mask_pos[0]
        ys = pix_deg*mask_pos[1]
        
        if mask_size == 0:
            mask[ np.where( np.sqrt((mx+1)**2 + (my+1)**2) > fsize/2) ] = 0
        elif mask_size <= diameter:
            mask[ np.where( np.sqrt((mx+1-xs)**2 + (my+1-ys)**2) > mask_size*pix_deg/2) ] = 0
        else:
            raise ValueError("Mask size (in degrees) must be less than stimulus diameter.")
    
    grating = contrast*0.5*np.cos( SF * xy + phi ) + 0.5
    
    # normalize the grating from zero to one:
    # if norm:
    #     grating = grating + np.abs(np.min(grating))
    #     grating = grating/np.max(grating)
    
    return grating * mask

def generate_ring_grating(theta, diameter, SF, pix_deg, contrast=1, phi=0, inner_d=1., outer_d=2., norm=True, mask_pos=[0,0]):
    fsize = pix_deg * diameter # pixels per size of img
    
    vals = np.linspace(-np.pi, np.pi, fsize)
    xv, yv = np.meshgrid(vals, vals)
    
    xy = xv*np.cos(theta) - yv*np.sin(theta)
    
    mask = np.ones((fsize, fsize))

    my,mx = np.ogrid[-fsize/2:fsize/2, -fsize/2:fsize/2] 
    
    xs = pix_deg*mask_pos[0]
    ys = pix_deg*mask_pos[1]
    
    if inner_d>=outer_d:
        raise ValueError("Inner diameter cannot be larger than the outer!")
    
    mask[ np.where( np.sqrt((mx+1-xs)**2 + (my+1-ys)**2) > outer_d*pix_deg/2) ] = 0

    mask[np.where(np.sqrt( (mx+1-xs)**2 + (my+1-ys)**2) < inner_d*pix_deg/2)] = 0
    
    grating = contrast*0.5*np.cos( SF * xy + phi ) + 0.5
    
    # normalize the grating from zero to one:
    # if norm:
    #     grating = grating + np.abs(np.min(grating))
    #     grating = grating/np.max(grating)
    
    return grating * mask



def generate_grating_bank(orientations, diameter, SF, pix_deg, phi=0, masked=False, norm=True):
    fsize = pix_deg*diameter
    bank = np.zeros((orientations.size,fsize,fsize))
    
    for i in range(orientations.size):
        bank[i,:,:] = generate_static_grating(orientations[i], diameter, SF, pix_deg, phi, masked, norm)
        
    return bank


# A class for a phenomenological V1 unit which generates the input to SSN E/I units
# These units have a gabor-type receptive field with phenomenological contrast normalization
# 
# Presently this implementation just uses the parameters found through nonlinear least-squares
# curve fitting performed in a separate Matlab script, the values of which are simply copied
# (except orientation preference which is taken from an orientation preference map)

class rf_unit:
    
    __default = object()
    
    def __init__(self, RF_size_deg, orient_pref, pix_deg=25, rf_size_pix=0):
        self.pix_per_deg = pix_deg 
        self.RF_size_deg = RF_size_deg
        self.ori_pref_deg = orient_pref
        
        if rf_size_pix == 0:
            self.RF_size_pix = np.round(pix_deg*RF_size_deg)
        else:
            self.RF_size_pix = rf_size_pix
            
        # self.sig_x = 1.5067
        # self.sig_y = 3.2409 
        # self.theta = orient_pref*np.pi/180
        # self.sf_pref = 0.1610
        # self.phi = 9.9233
        
        # self.gain = 30.4883
        # self.bias = -15.2404
        # self.rc_factor = 5.0

        # self.c_50 = 0.4149 
        # self.n = 1.0000
        # # self.r_max = 26.1949
        # self.r_max = 40.
        self.sig_x = 9.4277
        self.sig_y = 7.0001 
        self.theta = orient_pref*np.pi/180
        self.sf_pref = 0.0629
        self.phi = -5.6499
        
        self.gain = 3.1254
        self.bias = 1.9918
        self.rc_factor = 0.3687

        self.c_50 = 0.1635 
        self.n = 0.0033
        # self.r_max = 26.1949
        self.r_max = 40.

        self.RF = self.generate_RF()
        self.quad_RF = self.generate_RF(sine=True)
        
    def generate_RF(self, sine=False):
        return gabor(self.sig_x, self.sig_y, self.theta, self.sf_pref, self.phi, np.floor(self.RF_size_pix), sine ) 
    
    """
    Returns the RF (gabor) unit response to a static input image.
    The input image must be the same shape as the RF.
    The input image must also be previously normalized, with 0
    corresponding to black, and 1 to white values.
    """
    def get_unit_response_rate( self, input_img, neuron_RF=None ):
        gain = self.gain
        J_bias = self.bias
        if neuron_RF is None:
            neuron_RF = self.RF

        J = gain*np.sum( input_img * neuron_RF ) + J_bias
        if J < 0:
            return 0
        else:
            a = self.r_max*(J/(np.sqrt(J**2 + self.c_50**2)))**self.n;
            return a
    
    """ Quadrature pair response """
    def get_QP_response_rate( self, input_img ):
        # std_response = self.get_unit_response_rate( input_img )
        # qRF = self.quad_RF
        # quad_response = self.get_unit_response_rate( input_img, neuron_RF=qRF ) 
        # return np.sqrt( (quad_response)**2 + (std_response)**2 )
        gain = self.gain
        J_bias = self.bias

        J = gain*np.sum( np.sqrt( (input_img * self.RF)**2 + (input_img * self.quad_RF)**2 ) ) + J_bias
        if J < 0:
            return 0
        else:
            a = self.r_max*(J/(np.sqrt(J**2 + self.c_50**2)))**self.n;
            return a

    def get_LIF_response_rate( self, input_img ):

        tau_ref = 0.002
        gain = self.gain
        J_bias = self.bias
        tau_RC = self.rc_factor * 0.05
        neuron_RF = self.RF
        
        cos_sum = np.sum(input_img*neuron_RF)
        sin_sum = np.sum(input_img*self.quad_RF)

        J_unsat = gain*np.sqrt( cos_sum**2 + sin_sum**2 ) + J_bias
        c = np.max(input_img)-np.min(input_img)
        J = J_unsat * c**self.n/(self.c_50**self.n + c**self.n)
        
        j = J-1
        if j < 0.:
            return 0.,0.
        else:
            return 1. / ( tau_ref + tau_RC*np.log1p(1. / j)), J
            
    def show_RF(self):
        print "Showing RF!"
        plt.figure()
        plt.imshow(self.RF, cmap='gray')
        plt.colorbar()
        plt.title("Neuron RF")

        
class rf_layer:
    """
    Default constructor for a layer of RF (phenomenological) units for representing 
    static images as firing rates
    """
    def __init__(self, OP_map, N_pairs=75, field_size=16., uniform_rf_size=True, pix_deg=25):
        self.N_pairs = N_pairs
        self.field_size = field_size
        self.pix_deg = pix_deg
        
        if uniform_rf_size == True:
            self.rf_size_deg = field_size/N_pairs
        
        self.rf_size_pix = np.round(self.rf_size_deg*pix_deg)
        
        self.layer_units = []
        for i in range(self.N_pairs):
            self.layer_units.append([])
            for j in range(self.N_pairs):
                ori_pref = OP_map[i,j]
                new_unit = rf_unit(self.rf_size_deg, ori_pref, pix_deg=self.pix_deg, rf_size_pix=self.rf_size_pix)
                self.layer_units[i].append( new_unit )
                
    def get_layer_response_rates(self, input_img, phase_invariant=True ):
        if input_img.shape != (self.pix_deg*self.field_size, self.pix_deg*self.field_size):
            print "input image shape: ", input_img.shape
            print "Field size: ", self.pix_deg*self.field_size
            raise ValueError("Input image size does not match RF layer size.")
        
        layer_response = np.zeros((self.N_pairs, self.N_pairs))
        count = 0
        for i in range(self.N_pairs):
            for j in range(self.N_pairs):
                img_segment = input_img[j*self.rf_size_pix:j*self.rf_size_pix+self.rf_size_pix, i*self.rf_size_pix:i*self.rf_size_pix+self.rf_size_pix]
                if img_segment.shape != (self.rf_size_pix, self.rf_size_pix):
                    count = count+1
                else:
                    # if phase_invariant==True:
                    #     layer_response[j,i] = self.layer_units[j][i].get_QP_response_rate(img_segment)
                    # else:
                    #     layer_response[j,i] = self.layer_units[j][i].get_unit_response_rate(img_segment)
                    layer_response[j,i], _ = self.layer_units[j][i].get_LIF_response_rate( img_segment )
#         print "Skipped %d neurons." % count
        return layer_response
    
    def get_unit(self, y_ind, x_ind):
        return self.layer_units[y_ind][x_ind]
    
    def get_unit_response( self, input_img, unit_inds, show_segment=True, phase_invariant=False):
        xi = unit_inds[0]
        yi = unit_inds[1]
        
        img_segment = input_img[yi*self.rf_size_pix:yi*self.rf_size_pix+self.rf_size_pix, xi*self.rf_size_pix:xi*self.rf_size_pix+self.rf_size_pix]
        
        if show_segment==True:
            plt.figure()
            plt.imshow(img_segment, cmap='gray')
            plt.title("Input image segment")
            plt.colorbar()
            
            self.layer_units[yi][xi].show_RF()
        
        # if phase_invariant==True:
        #     response = self.layer_units[yi][xi].get_QP_response_rate(img_segment)
        # else:
        #     response = self.layer_units[yi][xi].get_unit_response_rate(img_segment)
        response = self.layer_units[yi][xi].get_LIF_response_rate(img_segment)
            
        return response