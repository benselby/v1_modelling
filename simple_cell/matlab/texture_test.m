% A test script for trying out pyramid-based texture reproduction on images
% of V1 topology
%
% Based on Heeger and Bergen, 1995


clear all
close all

input_im = imread('../figures/v1-topology-blasdel-figure6.png');
m = size(input_im,1);
n = size(input_im,2);

noise_im = wgn(m, n, 100);

%% Colour decorrelation on the input image
r_mean = mean2(input_im(:,:,1));
g_mean = mean2(input_im(:,:,2));
b_mean = mean2(input_im(:,:,3));

im_r = input_im(:,:,1) - r_mean;
im_g = input_im(:,:,2) - g_mean;
im_b = input_im(:,:,3) - b_mean;

% Calculate decorrelation transform M:
D = zeros(3,m*n);
D(1,:) = reshape(im_r, 1, m*n);
D(2,:) = reshape(im_g, 1, m*n);
D(3,:) = reshape(im_b, 1, m*n);

C = D*D';
[U,S,V] = svd(C);
M = S\U';

y = M*[im_r im_g im_b]; % colour-decorrelated image

%% Generate pyramidal subbands - using steerable pyramids


%% Histogram matching
% in_cdf = imhist(input_im);
% noise_cdf = imhist(noise_im);


%% Texture matching


%% Invert the colour transform


%% Show the results
figure()
subplot(1,2,1)
imshow(input_im)
title('Original Image from Blasdel, 1992')
subplot(1,2,2)
imshow(noise_im)
title('White Noise Image')