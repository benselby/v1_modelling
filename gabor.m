% Produces a square gabor filter of the specified fsize and with the
% specified parameters:
% sig_x, sig_y - scaling factors for the receptive field
% theta - orientation of the carrier (radians)
% phi - phase shift of the carrier (radians)
% k - the preferred spatial frequency 
% fsize - the fsize of one side of the filter - should be an odd number
% This function is to be used to produce filter banks representing macaque
% V1 orientation columns

function gb=gabor(sig_x, sig_y, theta, phi, k, fsize)

gb = zeros(fsize);
for i=1:fsize
   for j=1:fsize
       x = j - floor(fsize/2) - 1;
       y = i - floor(fsize/2) - 1;
       Xj = x*cos(theta) - y*sin(theta);
       Yj = x*sin(theta) + y*cos(theta);
       gb(i,j) = (1/(2*pi*sig_x*sig_y))*exp(-1*Xj^2/(2*sig_x^2) - Yj^2/(2*sig_y^2) )*cos(k*Xj-phi);
   end
end

end