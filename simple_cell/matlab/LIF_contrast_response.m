function a = LIF_contrast_response(params, input_data)
    sig_x = params(1);
    sig_y = params(2);
    theta = params(3);
    phi   = params(4);
    k     = params(5);
    gain  = params(6);
    J_bias    = params(7);
    RC_factor = params(8);
    sigma     = params(9);
    n         = params(10);
    
    global fsize;
    global contrast_bank;
    
    neuronRF = gabor(sig_x, sig_y, theta, phi, k, fsize);
    
    tau_ref = 0.002;
    tau_RC = 0.05*RC_factor;
    
    J = zeros(size(input_data,2), 1);
    c = zeros(size(input_data,2), 1);
    a = zeros(size(input_data,2), 1);
    
    for f=1:size(input_data,2)        
        J(f) = gain*sum(sum(neuronRF.*contrast_bank(:,:,f))) + J_bias; 
        % Contrast normalization
%         c = input_data(f);
        % Use Michelson contrast:
%         c = ( max(contrast_bank(:,:,f)) - min(contrast_bank(:,:,f)) ) / (max(contrast_bank(:,:,f)) + min(contrast_bank(:,:,f)));
        c(f) = max(max(contrast_bank(:,:,f))) - min(min(contrast_bank(:,:,f)));
%         J(f) = J(f)*c^n/(sigma^n + c^n);
        
    end    
    
%     J(J<1) = 0;
%     fr = 1./(tau_ref - tau_RC.*log(1 - (1./J)));
    J(J<0) = 0;
    fr = J;

    for i=1:size(input_data,2)
        a(i) = fr(i)*c(i)^n/(sigma^n + c(i)^n);
    end 
    a = fr;
    a=a';    
end