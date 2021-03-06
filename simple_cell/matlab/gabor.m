 % Produces a square gabor filter of the specified fsize and with the
% specified parameters:
% sig_x, sig_y - scaling factors for the receptive field
% theta - orientation of the carrier (radians)
% phi - phase shift of the carrier (radians)
% k - the preferred spatial frequency 
% fsize - the fsize of one side of the filter - should be an odd number
% This function is to be used to produce filter banks representing macaque
% V1 orientation columns

function gb=gabor(sig_x, sig_y, theta, phi, k, fsize, sine)

if nargin < 7
    sine = 0;
end

gb = zeros(fsize);
for i=1:fsize
   for j=1:fsize
       x = j - floor(fsize/2) - 1;
       y = i - floor(fsize/2) - 1;
       Xj = x*cos(theta) - y*sin(theta);
       Yj = x*sin(theta) + y*cos(theta);
       if sine==0
           gb(i,j) = (1/(2*pi*sig_x*sig_y))*exp(-1*Xj^2/(2*sig_x^2) - Yj^2/(2*sig_y^2) )*cos(k*Xj-phi);
       else
           gb(i,j) = (1/(2*pi*sig_x*sig_y))*exp(-1*Xj^2/(2*sig_x^2) - Yj^2/(2*sig_y^2) )*sin(k*Xj-phi);
       end
           
   end
end

%     sig_x = sig_x/fsize;
%     sig_y = sig_y/fsize;
%     vals = linspace(-pi, pi, fsize);
%     [xgrid, ygrid] = meshgrid(vals,vals);
%     xy = xgrid*cos(theta) + ygrid*sin(theta);
%     the_sine = sin(k*xy + phi);
%     the_gaussian = exp(-(xgrid/(2*sig_x^2))^2-(ygrid/(2*sig_y^2))^2);
%     gb = the_sine.*the_gaussian;
    
end