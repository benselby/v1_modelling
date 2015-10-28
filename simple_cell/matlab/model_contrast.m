clear all
close all

% Fixed Gabor filter parameters
global fsize
global contrast_bank

fsize = 25;
theta = pi/4;
k = 2;

%% Use Least-squares optimization 
%  to find optimal gabor and LIF parameters to fit the data

% rate     = [5    10   22   30  40  80  90 100 ];
% contrast = [0.03 0.06 0.15 0.2 0.3 0.5 0.7 1];

% Albrecht and Hamilton 1982 data, figure 1C:
data = [1.52,2.80;
        1.83,6.08;
        2.44,11.62;
        3.05,17.67;
        3.95,25.22;
        5.00,36.44;
        6.73,52.66;
        8.62,66.16;
        11.46,78.00;
        18.80,91.90;
        32.42,100.31;
        56.58,98.89];
    
contrast = data(:,1)';
rate = data(:,2)';

contrast = contrast/100;

contrast_bank = zeros(fsize, fsize, numel(contrast));
% grating = generate_grating(k, theta, fsize);
grating = generate_grating2(theta, 1, k, fsize);

for i=1:numel(contrast)
%     contrast_bank(:,:,i) = contrast(i)*(grating-0.5) + 0.5;
    contrast_bank(:,:,i) = contrast(i) * grating;
%     figure()
%     imshow(contrast_bank(:,:,i))
%     tit=sprintf('Contrast = %.2f', contrast(i));
%     title(tit)
end

figure()
imshow(contrast_bank(:,:,end))
title('Max. contrast stimulus used')

% Set lower and upper bounds for variables being optimized
% [sigx sigy theta phi k gain bias rc sigma n]
lb = [0.01 0.01   0   0    0.05 .0001 -100  1  0  1];
ub = [20   20   2*pi  2*pi 100  100    100  10 1 10]; 

numIter = 1;
xmulti = zeros(numIter,10);

paramsInitial = [1 1 pi/4 0 4 10 0 1 1 2];
problem = createOptimProblem('lsqcurvefit','x0',paramsInitial,'objective',@LIF_contrast_response,...
            'lb',lb,'ub',ub,'xdata',contrast,'ydata',rate);
% ms = MultiStart('PlotFcns',@gsplotbestf);
ms = MultiStart();

% ms.UseParallel = 'always';

[xmulti(1,:),errormulti] = run(ms,problem,100);   

neural_params = xmulti

estimate = LIF_contrast_response(neural_params, contrast);
figure();
plot(contrast, rate, 'bo-', contrast, estimate, 'ro-', 'LineWidth', 2)
title('Contrast Saturation', 'FontSize', 28)
xlabel('Contrast (%)', 'FontSize', 24)
ylabel('Response (Spikes / second)', 'FontSize', 24)
legend('Carandini et al. Data', 'Model Approximation', 'FontSize', 16)

fig = gcf;
fig.PaperUnits = 'inches';
fig.PaperPosition = [0 0 4.75 4.75];
fig.PaperPositionMode = 'manual';
print('matlab_figs/contrast_saturation','-depsc','-r0')

figure();
neuronRF = gabor(neural_params(1), neural_params(2), theta, neural_params(3), k, fsize);
imshow(mat2gray(neuronRF))
title('Receptive Field of Model Neuron')