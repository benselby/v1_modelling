function a = LIF_contrast_response(params, input_data)
    sig_x = params(1);
    sig_y = params(2);
    phi   = params(3);
    gain  = params(4);
    J_bias    = params(5);
    RC_factor = params(6);
    sigma     = params(7);
    n         = params(8);
    
    global fsize;
    global theta;
    global k;
    global contrast_bank;
    
    neuronRF = gabor(sig_x, sig_y, theta, phi, k, fsize);

    tau_ref = 0.002;
    tau_RC = 0.05*RC_factor;
    
    J = zeros(size(input_data,2), 1);
    
    for f=1:size(input_data,2)        
        J(f) = gain*sum(sum(neuronRF.*contrast_bank(:,:,f))) + J_bias; 
        % Contrast normalization
        c = input_data(f);
        J(f) = J(f)*c^n/(sigma^n + c^n);
    end    
   
    J(J<1) = 0;
    a = 1./(tau_ref - tau_RC.*log(1 - (1./J)));
    a=a';    
end