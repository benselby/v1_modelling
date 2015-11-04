function a = power_ori_contrast_response(params, input_data)
    sig_x = params(1);
    sig_y = params(2);
    theta = params(3);
    phi   = params(4);
    sf     = params(5);
    gain  = params(6);
    J_bias    = params(7);
    k = params(8);
    n = params(9);

    global stim_bank;
    neuronRF = gabor(sig_x, sig_y, theta, phi, sf, 25);
    quadRF = gabor(sig_x, sig_y, theta, phi, k, 25, 1);
    
    J = zeros(size(stim_bank,3), size(stim_bank,4));
    J_quad = zeros(size(stim_bank,3), size(stim_bank,4));
    
    for i=1:size(stim_bank,3)
        for j=1:size(stim_bank,4)
            J(i,j) = gain*sum(sum(neuronRF.*stim_bank(:,:,i,j))) + J_bias;
            J_quad(i,j) = gain*(sum(sum( quadRF.*stim_bank(:,:,i,j) ) ) ) + J_bias;
        end
    end    
    J(J<0) = 0;
    J_quad(J_quad<0) = 0;
    a1 = (k*J.^n).^2;
    a2 = (k*J_quad.^n).^2;
    a_sum = a1 + a2;
    a_sum(a_sum<0) = 0;
    a = sqrt( a_sum );
end