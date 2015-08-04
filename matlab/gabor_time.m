 % Produces a spatiotemporal gabor filter of the specified fsize and with the
% specified parameters:
% sig_x, sig_y - scaling factors for the receptive field
% theta - orientation of the carrier (radians)
% phi - phase shift of the carrier (radians)
% tf - temporal frequency of the RF(cycles per second)
% k - the preferred spatial frequency 
% fsize - the fsize of one side of the filter - should be an odd number
% T - length of receptive field in seconds
% dt - time discretization
% This function is to be used to produce individual gabor filters whi

function gb=gabor_time(sig_x, sig_y, theta, phi, tf, k, fsize, dt, T)

length = round(T/dt);
gb = zeros(fsize, fsize, length);
for t=1:length
    for i=1:fsize
       for j=1:fsize
           shift = 2*pi*tf*dt*(t-1); % spatiotemporal shift based on the temporal frequency
           x = j - floor(fsize/2) - 1;
           y = i - floor(fsize/2) - 1;
           Xj = x*cos(theta) - y*sin(theta) + shift*cos(theta);
           Yj = x*sin(theta) + y*cos(theta) + shift*sin(theta);
           gb(i,j,t) = (1/(2*pi*sig_x*sig_y))*exp(-1*Xj^2/(2*sig_x^2) - Yj^2/(2*sig_y^2) )*cos(k*Xj-phi);
       end
    end
end

end