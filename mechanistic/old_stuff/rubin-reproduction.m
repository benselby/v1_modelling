%
% Rereating the results from Rubin, Hooser, and Miller (2015)
% A stabilized supralinear network of interconnected E and I neurons 
% 
% Ben Selby, April 2015

clear all
close all

external_input = linspace(0,50);
k = 0.04;
n = 2;
firing_rate = k*external_input^n;
firing_rate(firing_rate < 0) = 0;

figure()
plot(external_input, firing_rate)
xlabel('Input')
ylabel('Output (Firing Rate)')
