function gaborArray = gaborFilterBank(u,v,m,n)

% GABORFILTERBANK generates a custum Gabor filter bank. 
% It creates a u by v array, whose elements are m by n matries; 
% each matrix being a 2-D Gabor filter.
% 
% 
% Inputs:
%       u	:	No. of scales (usually set to 5) 
%       v	:	No. of orientations (usually set to 8)
%       m	:	No. of rows in a 2-D Gabor filter (an odd integer number usually set to 39)
%       n	:	No. of columns in a 2-D Gabor filter (an odd integer number usually set to 39)
% 
% Output:
%       gaborArray: A u by v array, element of which are m by n 
%                   matries; each matrix being a 2-D Gabor filter   
% 
% 
% Sample use:
% 
% gaborArray = gaborFilterBank(5,8,39,39);
% 
% 
%   Details can be found in:
%   
%   M. Haghighat, S. Zonouz, M. Abdel-Mottaleb, "Identification Using 
%   Encrypted Biometrics," Computer Analysis of Images and Patterns, 
%   Springer Berlin Heidelberg, pp. 440-448, 2013.
% 
% 
% (C)	Mohammad Haghighat, University of Miami
%       haghighat@ieee.org
%       I WILL APPRECIATE IF YOU CITE OUR PAPER IN YOUR WORK.



if (nargin ~= 4)    % Check correct number of arguments
    error('There should be four inputs.')
end


%% Create Gabor filters

% Create u*v gabor filters each being an m*n matrix

gaborArray = cell(u,v);
fmax = 0.25;
gama = sqrt(2);
eta = sqrt(2);

% fus = [1.3090    9.1630   17.0170   24.8709   32.7249];

fus = [8 30 50 75 100] / 15.4;
% fus = round(fus);

sigmaX = 19*15;
sigmaY = 19*15;

halfWidth = n/2-0.5;
halfHeight = m/2-0.5;
x = repmat(-halfWidth:halfWidth, m, 1);
y = repmat((-halfHeight:halfHeight)', 1, n);
for i = 1:u
    
    fu = fmax/((sqrt(2))^(i-1));
    fus(i)= fu;
%     fu = fus(i);
    alpha = fu/gama;
    beta = fu/eta;
    
       
    for j = 1:v
        tetav = ((j-1)/v)*pi;
%         ((j-1)/v)*180
%         gFilter = zeros(m,n);
        
%         for x = 1:m
%             for y = 1:n
%                 xprime = (x-((m+1)/2))*cos(tetav)+(y-((n+1)/2))*sin(tetav);
%                 yprime = -(x-((m+1)/2))*sin(tetav)+(y-((n+1)/2))*cos(tetav);
        xprime = x*cos(tetav)-y*sin(tetav);
        yprime = x*sin(tetav)+y*cos(tetav);
        
        gFilter = (fu^2/(pi*gama*eta))*exp(-((alpha^2)*(xprime^2)+(beta^2)*(yprime^2)))*exp(1i*2*pi*fu*xprime);
%         gFilter  = (1/(2*pi*sigmaX*sigmaY)).*exp(-(xprime.^2 ./(2*sigmaX.^2))-(yprime.^2 ./(2*sigmaY.^2))).*exp(1i*2*pi*fu*xprime);
%             end
%         end
        gaborArray{i,j} = gFilter;
        
    end
end

%% Show Gabor filters

% Show magnitudes of Gabor filters:
% figure('NumberTitle','Off','Name','Magnitudes of Gabor filters');
% for i = 1:u
%     for j = 1:v        
%         subplot(u,v,(i-1)*v+j);        
%         imshow(abs(gaborArray{i,j}),[]);
%     end
% end

% Show real parts of Gabor filters:
figure('NumberTitle','Off','Name','Real parts of Gabor filters');
for i = 1:u
    for j = 1:v        
        subplot(u,v,(i-1)*v+j);        
        imshow(real(gaborArray{i,j}),[]);
    end
end
