classdef Rubin < handle
    % Utility functions for Rubin et al. (2015) full model. 
    
    methods (Static)
        
        function W_ab = getWeights(diff_pos, diff_angle, K_b, J_ab, sigma_ab, sigma_ori)
            % Finds sparse random weights between groups a and b (e.g.
            % excitatory and inhibitory, or excitatory and excitatory). 
            % 
            % diff_pos: pairwise position differences
            % diff_angle: pairwise preferred orientation differences
            % K_b: scale factor for connection probabilities
            % J_ab: mean weight
            % sigma_ab: width of gaussian connection probability vs.
            %   position difference
            % sigma_ori: width of gaussian connection probability vs.
            %   preferred orientation difference
            % 
            % W_ab: weight matrix for connection of b units to a units 
            
            p_ab = K_b * Rubin.G(diff_pos, sigma_ab) .* Rubin.G(diff_angle, sigma_ori); %connection probability
            C_ab = rand(size(p_ab)) < p_ab; % binary connections
            W_ab = J_ab + .25*J_ab*randn(size(p_ab));
            W_ab = W_ab .* C_ab;
            W_ab = max(0, W_ab);
            
            % From Rubin et al.: "Weights of a given type b onto each unit 
            % are then scaled so that all units of a given type a receive 
            % the same total type b synaptic weight, equal to Jab times the 
            % mean number of connections received ... ".

            % note: b (second index) refers to the origin
            scale = J_ab * mean(sum(C_ab, 2)) ./ sum(W_ab, 2);
            W_ab = W_ab .* repmat(scale, 1, size(W_ab, 2));
        end
        
        function d = angleDifference(x, y)
            % x: one orientation
            % y: another orientation 
            % 
            % d: difference between orientations (wrapped from 0 to 180)
            
            d = abs(mod(x - y + 90, 180) - 90);
        end
        
        function g = G(x, sigma) 
            % Unscaled gaussian function. 
            % 
            % x: independent variable 
            % sigma: the usual Gaussian width
            % 
            % g: unscaled zero-mean gaussian function of x
            
            g = exp(-x.^2 / 2 / sigma);
        end
        
    end
end