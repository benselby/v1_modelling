
clear all
close all

%Set global gabor filter parameter size
global fsize
fsize = 25;

%% Use Least-squares optimization 
%  to find optimal gabor and LIF parameters to fit the data
 
% rate        = [10  10  15  18  20  22  16  11  6   2  ];
% orientation = [360 354 349 345 340 335 330 326 321 318];

% Show the optimal input image:
figure()
% imshow( generate_bar(335, fsize) )
imshow( generate_grating(4, 335*pi/180, fsize) )
title('Preferred Stimulus')

rate        = [10  10  15  18  20  22  16  11  6   2   0   0   0   0   10  10  15  18  20  22  16  11  6   2   0   0  0  0];
orientation = [360 354 349 345 340 335 330 326 321 318 270 240 225 200 180 174 169 165 160 155 150 146 141 138 110 90 45 20];

% convert orientation to radians:
orient_r = orientation * pi/180;

% Scale the rate to max firing of 30 (for contour integration model params)
% rate = rate./max(rate)*130;

orient_norm = orientation / max(orientation);

% Create the bank of input images:
global input_bank;
input_bank = zeros(fsize, fsize, length(orientation));
for i=1:length(orientation)
%     input_bank(:,:,i) = generate_bar(orientation(i), fsize);
    input_bank(:,:,i) = generate_grating(4, orient_r(i), fsize);
end

% Set lower and upper bounds for variables being optimized
lb = [pi/10  pi/10  0    0    0.5 0.5 -100  1];   
ub = [20     20     2*pi 2*pi 10  100  100  10]; 

numIter = 1;
xmulti = zeros(numIter,8);

paramsInitial = [pi/4 pi/4 335*pi/180 0 4 0.15 50 1];
problem = createOptimProblem('lsqcurvefit','x0',paramsInitial,'objective',@LIF_Gabor_response,...
            'lb',lb,'ub',ub,'xdata',orientation,'ydata',rate);
% ms = MultiStart('PlotFcns',@gsplotbestf);
ms = MultiStart();
% ms.UseParallel = 'always'; % NOTE: Leave this out on Linux!!! will not run otherwise

[xmulti(1,:),errormulti] = run(ms,problem,100);    

neural_params = xmulti;

estimate = LIF_Gabor_response(neural_params, orientation);
figure();
plot(orientation, rate, 'o-', orientation, estimate, '*-')
title('V1 Simple Cell Orientation Tuning')
xlabel('Stimulus Orientation (Degrees)')
ylabel('Response (Spikes / second)')
legend('DeValois et al. Data', 'Model Approximation')

figure();
neuronRF = gabor(neural_params(1), neural_params(2), neural_params(3), neural_params(4), neural_params(5), fsize);
imshow(mat2gray(neuronRF))
title('Receptive Field of Model Neuron')

lb
ub
neural_params

