function a = LIF_contrast_response_no_gabor_no_lif(params, input_data)
    sigma     = params(1);
    n         = params(2);
    global gabor_rf;
    global lif_params;
    global contrast_bank;
    
    gain = lif_params(1);
    J_bias = lif_params(2);
    tau_RC = 0.05*lif_params(3);

%     gain = params(3);
%     J_bias = params(4);
%     tau_RC = 0.05*params(5);
    neuronRF = gabor_rf;
    tau_ref = 0.002;
    
    J = zeros(size(input_data,2), 1);
    c = zeros(size(input_data,2), 1);
    a = zeros(size(input_data,2), 1);
    
    for f=1:size(input_data,2)        
        J(f) = gain*sum(sum(neuronRF.*contrast_bank(:,:,f))) + J_bias; 
        % Contrast normalization
%         c = input_data(f);
        % Use Michelson contrast:
%         c(f) = ( max(contrast_bank(:,:,f)) - min(contrast_bank(:,:,f)) )
%         / (max(contrast_bank(:,:,f)) + min(contrast_bank(:,:,f)));
        c(f) = (max(max(contrast_bank(:,:,f))) - min(min(contrast_bank(:,:,f))));
%         J(f) = J(f)*c(f)^n/(sigma^n + c(f)^n);
    end    
    J(J<1) = 0;
    fr = 1./(tau_ref - tau_RC.*log(1 - (1./J)));
%     J(J<0) = 0;
%     fr = J;
    for i=1:size(input_data,2)
        a(i) = fr(i)*c(i)^n/(sigma^n + c(i)^n);
    end 
%     a = fr;
    a=a';    
end