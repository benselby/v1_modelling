clear all
close all

% Fixed Gabor filter parameters
global stim_bank

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
        32.42,98.89;
        56.58,100.0;
        100, 100];
    
contrast = data(:,1)';
rate_contrast = data(:,2)';
rate_fraction = rate_contrast/max(rate_contrast);
contrast = contrast/100;

orientation = 0:10:180;
rate_ori = 30*gaussian_ori(0, orientation, 32);
rate_ori = rate_ori-2*min(rate_ori);
rate_ori(rate_ori<0) = 0;

rate = zeros(numel(orientation), numel(contrast));

stim_bank = zeros(fsize, fsize, numel(orientation), numel(contrast));
% grating = generate_grating(k, theta, fsize);

for i=1:numel(orientation)
    for j=1:numel(contrast)
        stim_bank(:,:,i,j) = contrast(j)*generate_grating2(orientation(i)*pi/180, 1, 2, fsize); 
        rate(i,j) = rate_ori(i)*rate_fraction(j);
    end
end

figure()
plot(orientation, rate(:,end), orientation, rate(:,6), 'b--', orientation, rate(:,2), 'b--', orientation, rate(:,end-1), 'r--');

% Set lower and upper bounds for variables being optimized
% [sigx sigy theta phi sf gain bias k n c_gain]
lb = [1  2   0     0  1  .001   0  1  0 -100];
ub = [5  5  pi/180  1  10  500  2  20 3 0]; 

numIter = 1;
xmulti = zeros(numIter,10);

paramsInitial = [3.8831 4.4514 0 1.1430 5.2985 338.1044 .9992 50 2 -1];
problem = createOptimProblem('lsqcurvefit','x0',paramsInitial,'objective',@quad_power_ori_contrast_response,...
            'lb',lb,'ub',ub,'xdata',rate,'ydata',rate);
% ms = MultiStart('PlotFcns',@gsplotbestf);
ms = MultiStart();

% ms.UseParallel = 'always';

[xmulti(1,:),errormulti] = run(ms,problem,100);   

lb
ub
neural_params = xmulti

estimate = quad_power_ori_contrast_response(neural_params, contrast);
figure();
for j = 1:2:numel(contrast)
    plot(orientation, estimate(:,j), 'r', orientation, rate(:,j), 'b')
end

title('Contrast Saturation over Orientation', 'FontSize', 28)
xlabel('Orientation (degrees)', 'FontSize', 24)
ylabel('Response (Spikes / second)', 'FontSize', 24)
legend('Model Approximation', 'Data', 'FontSize', 16)
 
% fig = gcf;
% fig.PaperUnits = 'inches';
% fig.PaperPosition = [0 0 4.75 4.75];
% fig.PaperPositionMode = 'manual';
% print('matlab_figs/contrast_saturation','-depsc','-r0')

figure();
neuronRF = gabor(neural_params(1), neural_params(2), neural_params(3), neural_params(4), neural_params(5), fsize);
quadRF = gabor(neural_params(1), neural_params(2), neural_params(3), neural_params(4), neural_params(5), fsize,1);
subplot(1,2,1);
% imshow(mat2gray(neuronRF))
mesh(neuronRF)
title('Cosine RF')
subplot(1,2,2);
imshow(mat2gray(quadRF))
mesh(quadRF)
title('Quadrature RF')

figure()
mesh(contrast, orientation, estimate)
title('Model Response')

% figure()
% mesh(contrast, orientation, J_quad_ori_contrast_response(neural_params, contrast))
% title('Quadrature Current only')

% figure()
% mesh(contrast, orientation, J_power_ori_contrast_response(neural_params, contrast))
% title('Current only')
 
% figure()
% mesh(contrast, orientation, lif_unsat_ori_contrast_response(neural_params, contrast))
% title('LIF (unsaturated)')

figure()
mesh(contrast,orientation, rate)
title('Data')

% figure()
% best_params = [3.8831 4.4514 0 1.1430 5.2985 338.1044 .9992 5.9586 3.0159 2.0021];
% new_est = quad_ori_contrast_response(best_params, contrast);
% mesh(contrast, orientation, new_est)
% title('Best guess')