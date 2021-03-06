function a = LIF_Gabor_response(gabor_params, input_data)
    sig_x = gabor_params(1);
    sig_y = gabor_params(2);
    theta = gabor_params(3);
    phi   = gabor_params(4);
    k     = gabor_params(5);
    gain  = gabor_params(6);
    J_bias = gabor_params(7);
    RC_factor = gabor_params(8);
    
    global fsize;
    global input_bank;
    
    neuronRF = gabor(sig_x, sig_y, theta, phi, k, fsize);
    
    tau_ref = 0.002;
    tau_RC = 0.05*RC_factor;
    
    J = zeros(size(input_data,2), 1);
    for i=1:size(input_data,2)        
%         input_img = mat2gray(gabor(5, 7, input_data(i)*pi/180, 0, pi/8, fsize));    
        input_img = input_bank(:,:,i);
        J(i) = gain*sum(sum(neuronRF.*input_img)) + J_bias; 
    end      
    
%     J(J<0) = 0;
%     a=J';
     
    J(J<1) = 0;
    a = 1./(tau_ref - tau_RC.*log(1 - (1./J)));
    a=a';    
end