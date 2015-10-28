function a = LIF_sfreq_response(gabor_params, input_data)
    sig_x = gabor_params(1);
    sig_y = gabor_params(2);
    phi   = gabor_params(3);
    k     = gabor_params(4);
    gain  = gabor_params(5);
    J_bias = gabor_params(6);
    RC_factor = gabor_params(7);
    
    global fsize;
    global theta;
    global grating_bank;
    
    neuronRF = gabor(sig_x, sig_y, theta, phi, k, fsize);

    tau_ref = 0.002;
    tau_RC = 0.05*RC_factor;
    
    J = zeros(size(input_data,2), 1);
    
    for f=1:size(input_data,2)        
        J(f) = gain*sum(sum(neuronRF.*grating_bank(:,:,f))) + J_bias; 
    end
    
    
    J(J<1) = 0;
    a = 1./(tau_ref - tau_RC.*log(1 - (1./J)));
    a=a';    
end