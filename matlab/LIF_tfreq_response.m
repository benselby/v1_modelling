function a = LIF_tfreq_response(gabor_params, input_data)
    sig_x = gabor_params(1);
    sig_y = gabor_params(2);
    phi   = gabor_params(3);
    tf    = gabor_params(4);
    k     = gabor_params(5);
    gain  = gabor_params(6);
    J_bias = gabor_params(7);
    RC_factor = gabor_params(8);
    
    global fsize;
    global dt;
    global T;
    global theta;
    global grating_bank;
    
    neuronRF = gabor_time(sig_x, sig_y, theta, phi, tf, k, fsize, dt, T);

    tau_ref = 0.002;
    tau_RC = 0.05*RC_factor;
    
    J = zeros(size(input_data,1), 1);
    
    for f=1:size(input_data,1)   
        J(f) = gain*sum(sum(sum(neuronRF.*grating_bank(:,:,:,f)))) + J_bias; 
    end
    
    J(J<1) = 0;
    a = 1./(tau_ref - tau_RC.*log(1 - (1./J)));

end