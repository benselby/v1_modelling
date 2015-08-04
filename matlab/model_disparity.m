% A basic model for local disparity representation in macaque V1 
% Intended to reproduce figure 2 of Cumming and Parker (2000), which
% shows V1 simple cell responses to gratings of varying disparity

close all

% Values from figure xx of Cumming and Parker, 2000
rate      = [130   62   27   112  140 50  26  115 111]; % spikes per second
disparity = [-0.73 -0.6 -0.4 -0.2  0   0.2 0.4 0.6 0.75]; % degrees

% disparity = disparity*pi/180; % convert to radians

global fsize;
fsize = 101; % grating image size (pixels)

% Build the grating bank for regression - using parameters from paper
global grating_bank;
grating_freq = 4;
grating_theta = 0*pi/180;

grating_bank = zeros(fsize, fsize, 2*length(disparity)+1);
for i=1:length(disparity)
    phase = 2*pi*disparity(i)*grating_freq;
    grating_bank(:,:,2*i) = mat2gray(generate_grating(grating_freq, grating_theta, fsize));
    grating_bank(:,:,2*i+1) = mat2gray(generate_grating(grating_freq, grating_theta, fsize, phase));
%     figure
%     subplot(1,2,1)
%     imshow(grating_bank(:,:,2*i))
%     title_str = sprintf('Disp: %f', disparity(i));
%     title(title_str)
%     subplot(1,2,2)
%     imshow(grating_bank(:,:,2*i+1)) 
end

% figure
% subplot(1,3,1)
% im1 = mat2gray(generate_grating(grating_freq, grating_theta, fsize));
% imshow(im1)
% title_str = sprintf('Disp: %f', 0.73);
% subplot(1,3,2)    
% phase = 2*pi*.25/cos(theta)*grating_freq;
% im2 = mat2gray(generate_grating(grating_freq, grating_theta, fsize, phase));
% imshow(im2) 
% subplot(1,3,3)
% imshow(im1-im1)    
    
% Set lower and upper bounds for variables being optimized
% Variable order: [ sig_x sig_y theta phi_left phi_right k gain j_bias rc ]
lb = [2  2  0    0    0    1     0.05  -100  1];
ub = [100 100 pi pi/2 pi/2 2*pi*6  200   100  10]; 

numIter = 1;
xmulti = zeros(numIter,9);

paramsInitial = [10 10 0 0 0 2 1 50 1];
problem = createOptimProblem('lsqcurvefit','x0',paramsInitial,'objective',@LIF_disp_response,...
            'lb',lb,'ub',ub,'xdata', disparity, 'ydata',rate);
ms = MultiStart('PlotFcns',@gsplotbestf);
% ms.UseParallel = 'always';

[xmulti(1,:),errormulti] = run(ms,problem,100);    

neural_params = xmulti;

estimate = LIF_disp_response(neural_params, disparity );
figure();
plot(disparity, rate, 'ro-', disparity, estimate, '*-')
title('V1 Disparity Tuning')
xlabel('Dispartiy (degrees)')
ylabel('Response (Spikes / second)')
legend('Cumming and Parker Data', 'Model Approximation')

figure();
leftRF = gabor(neural_params(1), neural_params(2), neural_params(3), neural_params(4), neural_params(6), fsize);
rightRF = gabor(neural_params(1), neural_params(2), neural_params(3), neural_params(5), neural_params(6), fsize);
subplot(1,2,1)
imshow(mat2gray(leftRF))
subplot(1,2,2)
imshow(mat2gray(rightRF))
title('Receptive Fields of Model Neuron')

lb
ub
neural_params
