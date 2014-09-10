% A basic model for contour integration in macaque V1 
% Intended to reproduce figure 8a of Kapadia et al. (1995), which
% shows V1 cell facilitation in the presence of similarly oriented flanking
% bars.

close all

% Values from figure 8A of Kapadia, 1995
rate     = [58 50 44 32 36  29  22]; % spikes per second
sep_data = [32 49 67 84 100 116 135]; % separation (arc min)

% For figure D
% sep_data = [25 40 55 70 80 95 110]; % separation (arc min)
% rate     = [15 12 19 25 15 7  9]; % spikes per second

max_sep = max(sep_data);
    
% First create a RF with desired orientation and resting spike rate -
% parameters taken from model_orientation.m results
global fsize;
fsize = 25; % receptive field size (pixels)

global neuronRF;
global theta;
global k;
global phi;
theta = 0;
phi = 0;
k = 0.7225;
% k = 0.7853;
neuronRF = gabor(2, 125, theta, phi, k, fsize);

figure
imshow(mat2gray(neuronRF))
title('Classical Receptive Field')

global LIF_params;

LIF_params = [40.4469    0.8542    1.0000]; % For figure A
% LIF_params = [19.0195   -0.3441    2.4357]; % For figure D 

% Build the contour bank for regression:
global rf_spikes; % store the RF only spikes for plotting
global contour_bank;
global scale;
scale = 25; % pixels per degree

contour_bank = zeros(size(generate_contour(0, scale, max_sep/60, fsize)));
for i=1:length(sep_data)
    contour_bank(:,:,i) = generate_contour(sep_data(i)/60, scale, max_sep/60, fsize);
%     figure
%     imshow(contour_bank(:,:,i)) 
end

% Set lower and upper bounds for variables being optimized
% Variable order: [ sig_x sig_y theta phi k separation(degrees) weight sep2 weight2]
lb = [2  2       0      0.0001       0     -10];
ub = [10 100 max_sep/60  10      max_sep/60  10]; 

numIter = 1;
xmulti = zeros(numIter,6);

paramsInitial = [2 10 2 0.005 0.75 0.0025];
problem = createOptimProblem('lsqcurvefit','x0',paramsInitial,'objective',@LIF_contour_response,...
            'lb',lb,'ub',ub,'xdata', sep_data, 'ydata', rate);
ms = MultiStart('PlotFcns',@gsplotbestf);
ms.UseParallel = 'always';
[xmulti(1,:),errormulti] = run(ms,problem,100);    
neural_params = xmulti;

estimate = LIF_contour_response(neural_params, sep_data);
figure();
plot(sep_data, rate, 'ro-', sep_data, estimate, '*-', sep_data, rf_spikes, 'k-x')
title('V1 Coaxial Facilitation')
xlabel('Coaxial Separation (min. of arc)')
ylabel('Response (Spikes / second)')
legend('Kapadia et al. Data', 'Model Approximation')

% figure
% surroundRF = gabor(neural_params(1), neural_params(2), theta, phi, k, fsize);
% imshow(mat2gray(surroundRF));
% title('Nonclassical Receptive Field')

figure
totalRF = zeros( 2*fsize+round(max_sep/60*scale), fsize );
sep1 = neural_params(3);
w1   = neural_params(4);
sep2 = neural_params(5);
w2   = neural_params(6);
totalRF(1:fsize,:) = neuronRF; 
totalRF(fsize+round(sep1*scale)+1:2*fsize+round(sep1*scale), :) = imadd(totalRF(fsize+round(sep1*scale)+1:2*fsize+round(sep1*scale), :), w1*surroundRF);
totalRF(fsize+round(sep2*scale)+1:2*fsize+round(sep2*scale), :) = imadd(totalRF(fsize+round(sep2*scale)+1:2*fsize+round(sep2*scale), :), w2*surroundRF);
imshow(mat2gray(totalRF))
title('Total Receptive Field')

lb
ub
neural_params
