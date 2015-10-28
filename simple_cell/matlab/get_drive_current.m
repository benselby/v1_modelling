function J = get_drive_current(gabor_params, input_data, input_bank, fsize)
    sig_x = gabor_params(1);
    sig_y = gabor_params(2);
    theta = gabor_params(3);
    phi   = gabor_params(4);
    k     = gabor_params(5);
    gain  = gabor_params(6);
    J_bias = gabor_params(7);
    RC_factor = gabor_params(8);
    
    neuronRF = gabor(sig_x, sig_y, theta, phi, k, fsize);
    
    figure()
    imshow(mat2gray(neuronRF))
    
    J = zeros(size(input_data,2), 1);
    for i=1:size(input_data,2)        
%         input_img = mat2gray(gabor(5, 7, input_data(i)*pi/180, 0, pi/8, fsize));    
        input_img = input_bank(:,:,i);
        J(i) = gain*sum(sum(neuronRF.*input_img)) + J_bias; 
    end      
    
%     J(J<1) = 0;  
end