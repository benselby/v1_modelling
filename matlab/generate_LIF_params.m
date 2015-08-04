%
% A function for the v1_neuron class which generates LIF parameters which
% will produce the desired spike rate via some optimization
%

function [gain, j_bias, rc_factor] = generate_LIF_params(max_rate, orient_pref, bandwidth, RF)

rate = [0 max_rate 0];
orientation = [orient_pref-bandwidth orient_pref orient_pref+bandwidth];

fsize = size(RF,1);

% Generate response for optimization:
raw_resp = zeros(length(rate), 1);
for i=1:length(rate)
    input = generate_bar(orientation(i), fsize);
    raw_resp(i) = sum(sum(RF.*input));
end

lb = [0.1 -100  1];   
ub = [100  100  10]; 

numIter = 1;
xmulti = zeros(numIter,3);

paramsInitial = [0.15 50 1];
problem = createOptimProblem('lsqcurvefit','x0',paramsInitial,'objective',@LIF_param_response,...
            'lb',lb,'ub',ub,'xdata',raw_resp,'ydata',rate);
% ms = MultiStart('PlotFcns',@gsplotbestf);
ms = MultiStart();

[xmulti(1,:),errormulti] = run(ms,problem,50); 

gain = real(xmulti(1));
j_bias = real(xmulti(2));
rc_factor = real(xmulti(3));

% xmulti
% 
% resp = LIF_param_response([gain j_bias rc_factor], raw_resp);
% 
% figure
% plot(orientation, resp)

end