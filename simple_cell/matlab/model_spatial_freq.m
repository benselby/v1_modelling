clear all
close all

% Fixed Gabor filter parameters
global fsize
global theta
global grating_bank

fsize = 25;
theta = 0;

%% Use Least-squares optimization 
%  to find optimal gabor and LIF parameters to fit the data

rate = [3     3    3 5   10 11.5 33 28 11.5 10];
freq = [0.5   0.75 1 1.5 2  3    4  6  8    10];

grating_bank = zeros(fsize, fsize, numel(freq));
for i=1:numel(freq)
   grating_bank(:,:,i) = generate_grating(freq(i), theta, fsize);
   figure()
   imshow(grating_bank(:,:,i))
   tit=sprintf('Freq = %.2f', freq(i));
   title(tit)
end

% Set lower and upper bounds for variables being optimized
lb = [2  2  0    pi/20 0.5  -100  1];
ub = [10 10 2*pi pi    10    100  10]; 

numIter = 1;
xmulti = zeros(numIter,7);

paramsInitial = [0.5 0.05 0 pi/8 0.15 50 1];
% problem = createOptimProblem('lsqcurvefit','x0',paramsInitial,'objective',@LIF_sfreq_response,...
%             'lb',lb,'ub',ub,'xdata',freq,'ydata',rate);
problem = createOptimProblem('lsqcurvefit','x0',paramsInitial,'objective',@LIF_sfreq_response,...
            'xdata',freq,'ydata',rate);

ms = MultiStart('PlotFcns',@gsplotbestf);
% ms.UseParallel = 'always';

[xmulti(1,:),errormulti] = run(ms,problem,100);    

neural_params = xmulti

estimate = LIF_sfreq_response(neural_params, freq );
figure();
hold on
plot(freq, rate, 'bo-', 'LineWidth', 2)
plot(freq, estimate, 'ro-', 'LineWidth', 2)
title('Spatial Frequency Tuning', 'FontSize', 28)
xlabel('Spatial Frequency (cycles/deg)', 'FontSize', 24)
ylabel('Response (Spikes / second)', 'FontSize', 24)
legend('Foster et al. Data', 'Model Approximation', 'FontSize', 16)

fig = gcf;
fig.PaperUnits = 'inches';
fig.PaperPosition = [0 0 4.75 4.75];
fig.PaperPositionMode = 'manual';
print('matlab_figs/sf_tuning','-depsc','-r0')

figure();
neuronRF = gabor(neural_params(1), neural_params(2), theta, neural_params(3), neural_params(4), fsize);
imshow(mat2gray(neuronRF))
title('Receptive Field of Model Neuron')