clear all
close all

% Fixed Gabor filter parameters
global contrast_bank
global gabor_rf
global lif_params

fsize = 25;
% params = [9.8922 0.0131 6.2767 0.0006 98.7041 44.9080 -0.0398 4.8468];
% params = [2.2789    2.2775    5.4422    0.7661    8.4315   85.9627  -10.0236    1.1522];
% params = [2.9646    3.2894    1.1990    5.3254   20.2433    4.4404    0.9985    9.9511];
% params = [2.6606    2.9682    1.1948    4.9250   20.2988    3.2979    0.9955   10.0000];
params = [5.2208    2.1921    6.2801    1.5661    1.1282  145.4635    0.9971    9.9945];
gabor_rf = gabor(params(1),params(2),params(3),params(4),params(5), fsize);
lif_params = [params(6),params(7),params(8)];

figure()
imshow(mat2gray(gabor_rf))
title('RF to be used')

theta = params(3)+pi/2;
k = params(5);

%% Use Least-squares optimization 
%  to find optimal gabor and LIF parameters to fit the data
% rate     = [5    10   22   30  40  80  90  100 ];
% rate = rate/max(rate);
% rate = rate*130;
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
    
contrast = data(:,1)'/100;
rate = data(:,2)'/100;

contrast_bank = zeros(fsize, fsize, numel(contrast));
% grating = generate_grating(k, theta, fsize);
grating = generate_grating2(0, 1, 4, fsize);
c = zeros(size(contrast,2),1);

for i=1:numel(contrast)
%     contrast_bank(:,:,i) = contrast(i)*(grating-0.5) + 0.5;
    contrast_bank(:,:,i) = contrast(i)*grating;
%     figure()
%     imshow(contrast_bank(:,:,i))
%     tit=sprintf('Contrast = %.2f', contrast(i));
%     title(tit)
%     c(i) = ( max(max(contrast_bank(:,:,i))) - min(min(contrast_bank(:,:,i))) ) / (max(max(contrast_bank(:,:,i))) + min(min(contrast_bank(:,:,i))));
    c(i) = max(max(contrast_bank(:,:,i))) - min(min(contrast_bank(:,:,i)));
end

figure()
imshow(contrast_bank(:,:,end))
title('Max contrast grating')

% Set lower and upper bounds for variables being optimized
% [sigma n gain bias rc_factor] 
lb = [0  1 ];
ub = [1 10 ]; 

numIter = 1;
xmulti = zeros(numIter,2);

paramsInitial = [1 1];
problem = createOptimProblem('lsqcurvefit','x0',paramsInitial,'objective',@LIF_contrast_response_no_gabor_no_lif,...
            'lb',lb,'ub',ub,'xdata',contrast,'ydata',rate);
% ms = MultiStart('PlotFcns',@gsplotbestf);
ms = MultiStart();
% ms.UseParallel = 'always';

[xmulti(1,:),errormulti] = run(ms,problem,100);    

neural_params = xmulti

estimate = LIF_contrast_response_no_gabor_no_lif(neural_params, contrast);
figure();
plot(contrast, rate, 'bo-', contrast, estimate, 'ro-', 'LineWidth', 2)
title('Contrast Saturation', 'FontSize', 28)
xlabel('Contrast (%)', 'FontSize', 24)
ylabel('Response (%)', 'FontSize', 24)
legend('Carandini et al. Data', 'Model Approximation', 'FontSize', 16)

% fig = gcf;
% fig.PaperUnits = 'inches';
% fig.PaperPosition = [0 0 4.75 4.75];
% fig.PaperPositionMode = 'manual';
% print('matlab_figs/contrast_saturation','-depsc','-r0')

J = zeros(size(contrast,2),1);
tau_ref = 0.002;
tau_RC = 0.05*lif_params(3);
for i=1:numel(contrast)
    J(i) = lif_params(1)*sum(sum(contrast_bank(:,:,i).*gabor_rf)) + lif_params(2);
end
J(J<1) = 0;
a = 1./(tau_ref - tau_RC.*log(1 - (1./J)));
figure()
plot(contrast, a, 'r', contrast, J, 'b', contrast, rate, 'k--')
legend('Model Estimate', 'Current', 'Data')
title('Unsaturated activity')

