function grating = generate_grating2(theta, diameter, cycles_per_deg, pix_deg, phi, norm, masked, mask_size)

    if nargin < 5
        phi = 0;
    end
    if nargin < 6
        norm = 1;
    end
    if nargin < 7
        masked = 0;
    end
    if nargin < 8
        mask_size = 0;
    end
    
    fsize = pix_deg * diameter;
    
    vals = linspace(-pi, pi, fsize);
    [xv, yv] = meshgrid(vals, vals);
    
    xy = xv*cos(theta) + yv*sin(theta);
    
    mask = ones(fsize);
% 
%     
%     if masked == 1
%         if mask_size == 0
%             mask( find( sqrt((mx+1)^2 + (my+1)^2) > fsize/2) ) = 0;
%         elseif mask_size <= diameter
%             mask( find( sqrt((mx+1)^2 + (my+1)^2) > mask_size*pix_deg/2) ) = 0;
%         end
%     end
    cycles_per_frame=cycles_per_deg*diameter;
    grating = cos( cycles_per_frame * xy + phi ) .* mask;
    
    if norm == 1
        grating = mat2gray(grating);
    end
   
end