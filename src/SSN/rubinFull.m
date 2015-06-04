% Rubin et al. (2015) full model. 

load('orientation-map.mat')

rng(1) 

Dx = 16/75;
pos = (1:75) * Dx;

posx = repmat(pos, 75, 1);
posy = posx';

K_E = 0.1;
K_I = 0.5;
J_EE = 0.1;
J_IE = 0.38;
J_EI = 0.089;
J_II = 0.096;
k = 0.012;
n_E = 2.0;
n_I = 2.2;
sigma_EE = 8*Dx;
sigma_IE = 12*Dx;
sigma_EI = 4*Dx;
sigma_II = 4*Dx;
sigma_ori = 45; % degrees orientation
sigma_FF = 32; 
sigma_RF = Dx;

diff_angle = repmat(map(:), 1, 75^2) - repmat(map(:)', 75^2, 1);
diff_angle = abs(diff_angle);
diff_angle = min(diff_angle, 180-diff_angle);

diff_posx = repmat(posx(:), 1, 75^2) - repmat(posx(:)', 75^2, 1);
diff_posy = repmat(posy(:), 1, 75^2) - repmat(posy(:)', 75^2, 1);
diff_pos = (diff_posx.^2 + diff_posy.^2).^(1/2);
clear diff_posx diff_posy
%TODO: I think the positions are supposed to be periodic too 

W_EE = Rubin.getWeights(diff_pos, diff_angle, K_E, J_EE, sigma_EE, sigma_ori);
W_II = Rubin.getWeights(diff_pos, diff_angle, K_I, J_II, sigma_II, sigma_ori);
W_EI = Rubin.getWeights(diff_pos, diff_angle, K_I, J_EI, sigma_EI, sigma_ori);
W_IE = Rubin.getWeights(diff_pos, diff_angle, K_E, J_IE, sigma_IE, sigma_ori);

n = 2;

%TODO: no phase preferences, right?
stimulusRadius = 4;
stimulusMask = ((posx(:)-8).^2 + (posy(:)-8).^2).^.5 <= stimulusRadius;
h = Rubin.G(Rubin.angleDifference(map(:), 45), 30) .* stimulusMask;
c = 40;
% c = 50;

dt = .005;
steps = 100;
tau_ex = .02;
tau_in = .01;

I_E = zeros(75^2, steps+1);
I_I = zeros(75^2, steps+1);
r_E = zeros(75^2, steps+1);
r_I = zeros(75^2, steps+1);

for i = 1:steps
    fprintf('step %i of %i\n', i, steps)
    I_E(:,i+1) = c*h + W_EE * r_E(:,i) + W_EI * r_I(:,i);
    I_I(:,i+1) = c*h + W_IE * r_E(:,i) + W_II * r_I(:,i);

    rss_E = k*max(0, I_E(:,i)).^n; %steady-state rate
    rss_I = k*max(0, I_E(:,i)).^n;
    r_E(:,i+1) = r_E(:,i) + dt * (1/tau_ex) * (-r_E(:,i) + rss_E);
    r_I(:,i+1) = r_I(:,i) + dt * (1/tau_in) * (-r_I(:,i) + rss_I);    
end

figure
imagesc(reshape(r_E(:,end), 75, 75))
set(gca, 'FontSize', 18)
title('final spike rates', 'FontSize', 18)
colorbar
