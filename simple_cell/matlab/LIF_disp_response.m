function a = LIF_disp_response(params, input_data)
    sig_x = params(1);
    sig_y = params(2);
    theta = params(3);
    phi_left = params(4);
    phi_right = params(5);
    k     = params(6);
    gain  = params(7);
    J_bias = params(8);
    RC_factor = params(9);
    
    global fsize;
    global grating_bank;
    
    left_RF = gabor(sig_x, sig_y, theta, phi_left, k, fsize);
    right_RF = gabor(sig_x, sig_y, theta, phi_right, k, fsize);
    
    tau_ref = 0.002;
    tau_RC = 0.05*RC_factor;
    
    J = zeros(size(input_data,2), 1);
    for i=1:size(input_data,2)        
        left_img  = grating_bank(:,:,2*i);
        right_img = grating_bank(:,:,2*i+1);
        
        total_drive = sum(sum(left_RF.*left_img)) + sum(sum(right_RF.*right_img));
        
        J(i) = gain*total_drive + J_bias; 
    end      
    
    % LIF nonlinearity
    J(J<1) = 0;
    a = 1./(tau_ref - tau_RC.*log(1 - (1./J)));
    a=a';    
end