%
% A function for the v1_neuron class which generates a gabor RF based on
% the specified data 
%

function RF = generate_RF(bandwidth, orient_pref, fsize)
    sig_x = bandwidth/360*fsize; % some function of fsize and the bandwidth
    sig_y = bandwidth/360*fsize;
    theta = orient_pref * pi / 180;
    phi = 0; % some fxn of the disparity...
    k = 0.5; % some fxn of spatial frequency preference

    RF = gabor(sig_x, sig_y, theta-pi/2, phi, k, fsize);
%     imshow( mat2gray( RF ))
end