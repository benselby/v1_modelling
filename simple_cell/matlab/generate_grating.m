% Generate a sinusoidal grating
function gr=generate_grating(freq, theta, fsize, phase)

    if nargin < 4
        phase = 0;
    end
    
    X = 1:fsize;                           % X is a vector from 1 to imageSize
    X0 = (X / fsize) - .5; 
    [Xm, Ym] = meshgrid(X0, X0); 

    Xt = Xm * cos(theta);                % compute proportion of Xm for given orientation
    Yt = Ym * sin(theta);                % compute proportion of Ym for given orientation
    XYt = Xt + Yt;                      % sum X and Y components
    XYf = XYt * freq * 2*pi;                % convert to radians and scale by frequency
    gr = sin( XYf + phase );                   % make 2D sinewave
    
%     gr = zeros(fsize);
%     for i=1:fsize
%        for j=1:fsize
%            x = i - floor(fsize/2) - 1;
%            y = j - floor(fsize/2) - 1;
%            Xj = x*cos(theta) - y*sin(theta);
%            gr(i,j) = mat2gray(cos(freq*(2*pi)*Xj));  
%        end
%     end 
end