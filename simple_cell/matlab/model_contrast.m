clear all
close all

% Fixed Gabor filter parameters
global fsize
global theta
global contrast_bank
global k

fsize = 25;
theta = pi/4;
k = 2;

%% Use Least-squares optimization 
%  to find optimal gabor and LIF parameters to fit the data

rate     = [5    10   22   30  40  80  90  100 ];
contrast = [0.03 0.06 0.15 0.2 0.3 0.5 0.7 1];

contrast_bank = zeros(fsize, fsize, numel(contrast));
grating = generate_grating(k, theta, fsize);
for i=1:numel(contrast)
    contrast_bank(:,:,i) = contrast(i)*(grating-0.5) + 0.5;
%     figure()
%     imshow(contrast_bank(:,:,i))
%     tit=sprintf('Contrast = %.2f', contrast(i));
%     title(tit)
end

% Set lower and upper bounds for variables being optimized
lb = [2  2   0     0.05  -100  1  1  0];
ub = [10 10  pi/2  10     100  10 10 3]; 

numIter = 1;
xmulti = zeros(numIter,8);

paramsInitial = [0.5 0.05 0 0.15 50 1 1 1];
problem = createOptimProblem('lsqcurvefit','x0',paramsInitial,'objective',@LIF_contrast_response,...
            'lb',lb,'ub',ub,'xdata',contrast,'ydata',rate);
ms = MultiStart('PlotFcns',@gsplotbestf);
ms.UseParallel = 'always';

[xmulti(1,:),errormulti] = run(ms,problem,100);    

neural_params = xmulti

estimate = LIF_contrast_response(neural_params, contrast);
figure();
plot(contrast, rate, 'ro-', contrast, estimate, '*-')
title('V1 Contrast Tuning')
xlabel('Contrast (%)')
ylabel('Response (Spikes / second)')
legend('Carandini et al. Data', 'Model Approximation')

figure();
neuronRF = gabor(neural_params(1), neural_params(2), theta, neural_params(3), k, fsize);
imshow(mat2gray(neuronRF))
title('Receptive Field of Model Neuron')