% Function which produces the external input shape h(x) for the ring model
% described in Rubin et al., 2015

function h = input_shape_ring(phi, theta)

% phi: stimulus orientation
% theta: range of possible orientations

dxy = abs(phi-theta);
sig_FF = 30; % from suppemental methods

h = exp(-1*dxy.^2 / (2*sig_FF^2));

end