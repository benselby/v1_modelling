function J = J_power_ori_contrast_response(params, input_data)
    sig_x = params(1);
    sig_y = params(2);
    theta = params(3);
    phi   = params(4);
    sf     = params(5);
    gain  = params(6);
    J_bias    = params(7);
    k = params(8);
    n = params(9);
    c_gain = params(10);

    global stim_bank;
    neuronRF = gabor(sig_x, sig_y, theta, phi, sf, 25);
    
    J = zeros(size(stim_bank,3), size(stim_bank,4));
    a = zeros(size(stim_bank,3), size(stim_bank,4));
    c = zeros(size(stim_bank,4),1);
    
    for j=1:size(stim_bank,4)
        c(j) = (max(max(stim_bank(:,:,1,j))) - min(min(stim_bank(:,:,1,j))));
    end
    
    for i=1:size(stim_bank,3)
        for j=1:size(stim_bank,4)
%             J(i,j) = gain*sum(sum(neuronRF.*stim_bank(:,:,i,j))) + J_bias; 
%             J(i,j) = sum(sum(neuronRF.*stim_bank(:,:,i,j))); 
            J(i,j) = gain*sum(sum(neuronRF.*stim_bank(:,:,i,j))) + J_bias; 
        end
    end    
%     J(J<0) = 0;
   
end