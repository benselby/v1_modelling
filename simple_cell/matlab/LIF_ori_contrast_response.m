function a = LIF_ori_contrast_response(params, input_data)
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

    global stim_bank;
    
    tau_RC = 0.05*RC_factor;
    
    neuronRF = gabor(sig_x, sig_y, theta, phi, k, 25);
    tau_ref = 0.002;
    
    J = zeros(size(stim_bank,3), size(stim_bank,4));
    c = zeros(size(stim_bank,4),1);
    
    for j=1:size(stim_bank,4)
        c(j) = (max(max(stim_bank(:,:,1,j))) - min(min(stim_bank(:,:,1,j))));
    end
    
    for i=1:size(stim_bank,3)
        for j=1:size(stim_bank,4)
            J(i,j) = gain*sum(sum(neuronRF.*stim_bank(:,:,i,j))) + J_bias; 
        end
    end    
    J(J<1) = 0;
    fr = 1./(tau_ref - tau_RC.*log(1 - (1./J)));
%     J(J<0) = 0;
%     fr = J;
%     for i=1:size(stim_bank,3)
%         for j=size(stim_bank,4)
% %             a(i,j) = (fr(i,j))*c(j)^n/(sigma^n + c(j)^n);
%             a(i,j) = fr(i,j)^n/(sigma^n + fr(i,j)^n);
%         end
%     end     
%     a = max(max(fr))*fr.^n ./ (sigma^n + fr.^n);
    a = 30*fr.^n ./ (sigma^n + fr.^n);
end