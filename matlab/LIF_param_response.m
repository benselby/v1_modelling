function a = LIF_param_response(LIF_params, input_data)
    gain = LIF_params(1);
    j_bias = LIF_params(2);
    rc_factor = LIF_params(3);

    tau_ref = 0.002;
    tau_RC = 0.05*rc_factor;

    J = gain*input_data + j_bias;
    J(J<0) = 0;
    a = 1./(tau_ref - tau_RC.*log(1-(1./J)));
    a=a';
end