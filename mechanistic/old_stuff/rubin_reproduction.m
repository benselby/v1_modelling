%
% Rereating the results from Rubin, Hooser, and Miller (2015)
% A stabilized supralinear network of interconnected E and I neurons 
% 
% Ben Selby, April 2015

clear all
close all

%% Ring Model - described on pg 404

% Parameters - from the supplemental methods 
tau_E = 0.02; % 20 ms
tau_I = 0.01;

N = 180; % total number of neurons
theta = 1:180; % orientation range (degrees)

J_EE = 0.044;
J_IE = 0.042;
J_EI = 0.023;
J_II = 0.018;

sig_ori = 32; % preferred stimulus orientation (degrees)

k = 0.04;
n = 2;

% develop the external input h(x):
h = input_shape_ring(sig_ori, theta);
figure()
plot(theta, h)
title('Input shape h(x) for ring model')

% develop input to each neuron:
c = 50; % contrast value for input stimulus - from figure 1 caption
I_E = c*h

external_input = linspace(0,50);
firing_rate = k*external_input.^n;
firing_rate(firing_rate < 0) = 0;
figure()
title('Power-law input/out for a neuron')
plot(external_input, firing_rate)
xlabel('Input')
ylabel('Output (Firing Rate)')
