% A function for generating images made up of two bars for contour
% integration models based on Kapadia et al., 1995 - currently generates
% vertical coaxial line segments only
% Parameters: 
% separation: of line segment (degrees)
% scale: pixels per degree horizontal/vertical
% max_sep: the maximum allowable separation of the two bars (degrees)
% fsize: the size of one bar/receptive field (pixels)
% theta: orientation of the bars
function img = generate_contour(separation, scale, max_sep, fsize, theta)
    if nargin < 5
        theta = 0;
    end
    img = zeros(2*fsize+round(max_sep*scale), fsize);
    img(1:fsize, 12:14) = 1;
    img(fsize+round(separation*scale):2*fsize+round(separation*scale), 12:14) = 1;
    
    img = imrotate(img, theta);
end