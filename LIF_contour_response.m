function a = LIF_contour_response(params, input_data)
    sig_x = params(1);
    sig_y = params(2);
    sep   = params(3);
    w1    = params(4);
    sep2  = params(5);
    w2    = params(6);
    
    global fsize;
    global contour_bank;
    global neuronRF;    
    global scale;
    global LIF_params;
    global theta;
    global k;
    global phi;
    global rf_spikes;
    
    gain = LIF_params(1);
    J_bias = LIF_params(2);
    RC_factor = LIF_params(3);
    
    tau_ref = 0.002;
    tau_RC = 0.05*RC_factor;
    
    surroundRF = gabor(sig_x, sig_y, theta, phi, k, fsize);
    surround_range = fsize+round(sep*scale)+1:2*fsize+round(sep*scale);
    s2_range = fsize+round(sep2*scale)+1:2*fsize+round(sep2*scale);
    
    J = zeros(size(input_data,2), 1);
    J_rf = zeros(size(input_data,2), 1);
    for i=1:size(input_data,2)        
        input_img = contour_bank(:,:,i);   
        rf_drive = sum(sum(neuronRF.*input_img(1:fsize, 1:fsize) ) );
        surround_drive = sum(sum(surroundRF.*input_img(surround_range, :) ) );
        s2_drive = sum(sum(surroundRF.*input_img(s2_range, :) ) );
        J(i) = gain*(rf_drive + w1*surround_drive + w2*s2_drive) + J_bias; 
        J_rf(i) = gain*(rf_drive) + J_bias; 
    end      
    
    J(J<1) = 0;
    a = 1./(tau_ref - tau_RC.*log(1 - (1./J)));
    a=a';    
    
    
    J_rf(J_rf<1) = 0;
    rf_spikes = 1./(tau_ref - tau_RC.*log(1 - (1./J_rf)));
    rf_spikes=rf_spikes ';
    
end