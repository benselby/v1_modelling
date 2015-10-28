
clear all
close all

%Set global gabor filter parameter size
global fsize
fsize = 25;

%% Use Least-squares optimization 
%  to find optimal gabor and LIF parameters to fit the data
 
rate        = [10  10  15  18  20  22  16  11  6   2  ];
orientation = [360 354 349 345 340 335 330 326 321 318];

% Show the optimal input image:
% figure()
bar = generate_bar(335, fsize);
% imshow( bar )

rate        = [10  10  15  18  20  22  16  11  6   2   0   0   0   0   10  10  15  18  20  22  16  11  6   2   0   0  0  0];
orientation = [360 354 349 345 340 335 330 326 321 318 270 240 225 200 180 174 169 165 160 155 150 146 141 138 110 90 45 20];

orientation = 0:180;
rate = gaussian_ori(0, orientation, 32);

% Scale the rate to max firing of 30 (for contour integration model params)
rate = rate./max(rate);
% min(rate)
% rate(rate<.02) = 0;

orient_norm = orientation / max(orientation);

% Create the bank of input images:
global input_bank;
input_bank = zeros(fsize, fsize, length(orientation));
% for i=1:length(orientation)
%     input_bank(:,:,i) = generate_bar(orientation(i), fsize);
% end
for i=1:length(orientation)
    ori_rad = orientation(i) * pi / 180;
    input_bank(:,:,i) = generate_grating2(ori_rad, 1, 4, fsize);
end

figure()
imshow(input_bank(:,:,1))

figure()
imshow(input_bank(:,:,30))

figure()
imshow(input_bank(:,:,50))

% Set lower and upper bounds for variables being optimized
lb = [.01 .01  0    0   1    0.01 -100   1];   
ub = [20  20 2*pi 2*pi  10  200   100 10]; 

numIter = 1;
xmulti = zeros(numIter,8);

% paramsInitial = [3 3 0 0 4 1 50 1];
paramsInitial = [2.9646    3.2894    1.1990    5.3254   5    4.4404    0.9985    9.9511];
problem = createOptimProblem('lsqcurvefit','x0',paramsInitial,'objective',@LIF_Gabor_response,...
            'lb',lb,'ub',ub,'xdata',orientation,'ydata',rate);

% problem = createOptimProblem('lsqcurvefit','x0',paramsInitial,'objective',@LIF_Gabor_response,...
%             'xdata',orientation,'ydata',rate);

% ms = MultiStart('PlotFcns',@gsplotbestf);
ms = MultiStart();
% ms.UseParallel = 'always'; % NOTE: Leave this out on Linux!!! will not run otherwise

[xmulti(1,:),errormulti] = run(ms,problem,100);    

neural_params = xmulti;

estimate = LIF_Gabor_response(neural_params, orientation);
figure();
hold on
plot(orientation, rate, 'bo-', 'LineWidth', 2)
plot(orientation, estimate, 'ro-', 'LineWidth', 2)

title('Orientation Tuning', 'FontSize', 28)
xlabel('Stimulus Orientation (Degrees)', 'FontSize', 24)
ylabel('Response (Spikes / second)', 'FontSize', 24)
legend('DeValois et al. Data', 'Model Approximation', 'FontSize', 16)

% fig = gcf;
% fig.PaperUnits = 'inches';
% fig.PaperPosition = [0 0 4.75 4.75];
% fig.PaperPositionMode = 'manual';
% print('matlab_figs/orientation_tuning','-depsc','-r0')

figure();
neuronRF = gabor(neural_params(1), neural_params(2), neural_params(3), neural_params(4), neural_params(5), fsize);
imshow(mat2gray(neuronRF))
title('Receptive Field of Model Neuron')

lb
ub
neural_params

