% get contrast 

% contrast = getContrast(mtNeuron, mapframe)
           
img = imread('E:\Visual Cortex Model\Optical flow\cvpr10_flow_code\flow_code_v2\2011_09_26\2011_09_26_drive_0017_sync\image_02\data\0000000000.png');

img = rgb2gray(img);

imSize = size(img);

figure(), imshow(img)%imshow(im(:,:,3))

% x = (-log(.5) / 2 / pi^2)^.5; 
% sigmat = x * 1 / max(prefFreqs) / sin(pi/2/length(prefOrientations)); % for highest spatial frequency
        
halfWidth = imSize(2)/2-0.5;
halfHeight =imSize(1)/2-0.5;

x = repmat(-halfWidth:halfWidth, imSize(1), 1);
y = repmat((-halfHeight:halfHeight)', 1, imSize(2));

            
% kernelsCos = zeros(imSize(1), imSize(2), length(prefFreqs), length(prefOrientations));
% kernelsSin = zeros(imSize(1), imSize(2), length(prefFreqs), length(prefOrientations));

st =    5; % sigmat * max(prefFreqs)/prefFreqs(i);

sigmar = 20;
prefOrientations = pi/180 * 90;
prefFreqs = 5;

a = prefOrientations;

xj = x*cos(a) - y*sin(a);
yj = x*sin(a) + y*cos(a);

cosine = cos(2*pi*prefFreqs*xj);
sine = sin(2*pi*prefFreqs*xj);

gaussian = 1 / (2*pi*sigmar*sigmar) * exp( 1/2 * (- (xj/sigmar).^2 - (yj/sigmar).^2) );
% gaussian = 1  * exp( 1/2 * (- (xj/sigmar).^2 - (yj/sigmar).^2) );

% figure(), mesh(gaussian);

size(gaussian)

[row, col] = find(gaussian > .0001);

gaussian = gaussian(min(row):max(row), min(col):max(col));
cosine = cosine(min(row):max(row), min(col):max(col));
sine = sine(min(row):max(row), min(col):max(col));

size(gaussian)

% figure(), mesh(gaussian);
kernelsCos = gaussian .* cosine;
kernelsSin = gaussian .* sine;

figure(), subplot(2,1,1), mesh(kernelsCos);
subplot(2,1,2), mesh(kernelsSin);

                   
filtImgCos = conv2(double(img),kernelsCos,'same');
filtImgSin = conv2(double(img),kernelsSin,'same');

filtImg = (filtImgSin.^2 + filtImgCos.^2).^0.5;

figure(), imshow(uint8(filtImg));



% contrast  = mean (mapframe(:));