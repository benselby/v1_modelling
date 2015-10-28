% A receptive field model for temporal frequency in V1 neurons,
% based on figure 8 of Foster et al., 1985.
%
% Ben Selby, 2014

clear all
close all

%% Data from figure 8 of Foster et al., 1985 - for reproduction
% Extracted using WebPlotDigitizer
% (arohatgi.info/WebPlotDigitizer/index.html)

% Figure A
data = [0.50219, 8.0021;
0.71667, 9.0784;
0.99014, 8.8791;
1.4360, 10.735;
1.9576, 12.708;
2.7947, 14.884;
4.0683, 25.536;
5.7172, 28.973;
7.9516, 12.654;
11.072, 6.1451;
15.782, 6.2369];

% % Figure B
% data = [0.702, 46.192;
% 0.992, 44.653;
% 1.425, 53.269;
% 2.000, 59.242;
% 2.807, 78.501;
% 4.062, 95.305;
% 5.613, 74.220;
% 8.000, 58.485;
% 11.227, 35.640;
% 15.635, 23.432];
% 
% % Figure C
% data = [0.509, 49.945;
% 0.720, 41.183;
% 1.006, 47.170;
% 1.454, 38.481;
% 2.045, 40.218;
% 2.830, 40.259;
% 4.130, 36.188;
% 5.845, 33.595;
% 8.181, 23.439;
% 11.807, 16.006;
% 16.268, 10.870];

rate = data(:,2); % spikes per second
freq = data(:,1); % cycles per second

global T;
global dt;
global fsize;
global theta;
global grating_bank;

T = 1; % duration of stimuli(seconds)
dt = 1/(2*ceil(max(freq))); % time discretization
fsize = 25; % size of the stimuli (pixels to a side)
sfreq = 4; % spatial frequency of stimuli
theta = pi/3; % orientation of stimuli

%% Generate drifting gratings - single (preferred) spatial frequency, 
% varying temporal (drifting) frequencies
grating_bank = zeros(fsize, fsize, round(T/dt), length(freq));

for i=1:length(freq)
   grating_bank(:,:,:,i) = generate_drift_grating(T, dt, sfreq, freq(i), theta, fsize); 
end

% %% Play an animation of one of the gratings
% gr = grating_bank(:,:,:,10);
% for i=1:size(gr,3)
%     imshow(gr(:,:,i))
%     drawnow
%     pause(dt)
% end

%%%%%%%%%%%%%%%%%%%%%
gabor_bank = zeros(fsize, fsize, round(T/dt));

for i=1:length(freq)
  gabor_bank = gabor_time(5, 5, pi/4, 0, 2, 4, fsize, dt, T);
end

% %% Play an animation of one of the gratings
% for i=1:size(gabor_bank,3)
%     imshow(mat2gray(gabor_bank(:,:,i)))
%     drawnow
%     pause(dt)
% end

% Set lower and upper bounds for variables being optimized
% [sig_x sig_y phi tf k gain j_bias RC_factor]
lb = [2  2  pi/20 0  0  0.5  -100  1];
ub = [10 10 pi    15 20 10    100  10]; 

numIter = 1;
xmulti = zeros(numIter,8);

paramsInitial = [5 5 0 2 pi/8 1 0 1];

problem = createOptimProblem('lsqcurvefit','x0',paramsInitial,'objective',@LIF_tfreq_response,...
            'lb',lb,'ub',ub,'xdata',freq,'ydata',rate);
        
ms = MultiStart('PlotFcns',@gsplotbestf);
% ms.UseParallel = 'always';

[xmulti(1,:),errormulti] = run(ms,problem,100);    

neural_params = xmulti

estimate = LIF_tfreq_response(neural_params, freq);
figure();
plot(freq, rate, 'bo-', freq, estimate, 'ro-', 'LineWidth', 2)
title('Temporal Frequency Tuning', 'FontSize', 28)
xlabel('Temporal Frequency (cycles/sec)', 'FontSize', 24)
ylabel('Response (Spikes / second)', 'FontSize', 24)
legend('Foster et al. Data', 'Model Approximation', 'FontSize', 16)

fig = gcf;
fig.PaperUnits = 'inches';
fig.PaperPosition = [0 0 4.75 4.75];
fig.PaperPositionMode = 'manual';
print('matlab_figs/tf_tuning','-depsc','-r0')

figure();
neuronRF = gabor_time(neural_params(1), neural_params(2), theta, neural_params(3), neural_params(4), neural_params(5), fsize, dt, T);

mesh(reshape( neuronRF(:,12,:), 25, 32 ) )
title('Spatiotemporal Receptive Field of Model Neuron')
xlabel('Spatial X')
ylabel('Spatial Y')
zlabel('Time')

%% Play an animation of the neuron RF
for i=1:size(neuronRF,3)
    imshow(mat2gray(neuronRF(:,:,i)))
    drawnow
    pause(dt)
end

figure();
plot(freq, rate)
xlabel('Temporal Frequency (Hz)')
ylabel('Response (spikes/s)')
legend('Foster et al. Data')

