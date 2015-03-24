classdef NetworkPlot < handle
    
    methods (Static)
        function plotLayers(xres, yres, chan, level)
            % xres: list of #pixels in horizontal direction for each layer
            % yres: list of #pixels in vertical direction for each layer
            % chan: list of #channels for each layer
            % level: an integer category from 1 (V1) up in visual hierarchy
            
            levels = unique(level);
            levelOffsets = levels*100;
            for i = 1:length(levels)
                ind = find(level == levels(i));
                ty = sum(yres(ind));
                gap = 50;
                tyWithGaps = ty + gap*(length(ind)-1);
                bottom = -tyWithGaps/2;
                for j = 1:length(ind)
                    c = chan(ind(j)); x = xres(ind(j)); y = yres(ind(j));
                    NetworkPlot.plotBox([levelOffsets(i); bottom+y/2; 0], [10*log(c); y; x]);
                    bottom = bottom + 50 + y;
                end
            end
            axis equal
        end
        
        function plotBox(centre, size)
            % centre: [x;y;z] of box centre
            % size: [x;y;z] box dimensions
            
            hold on
            corners = repmat(centre,1,8) + repmat(size/2,1,8).*(-1).^ceil([4;2;1].^-1*(1:8)); % <-- I'm on *fire* 
            for i = 1:8
                for j = 1:8
                    if sum(corners(:,i) == corners(:,j)) == 2
                        c = corners(:,[i j]);
                        plot3(c(1,:), c(2,:), c(3,:), 'k');
                    end
                end
            end
            hold off
        end
    end

end