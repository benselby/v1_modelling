% Sketches a convolutional network based on visual cortex regions from 
% Markov et al. 

% im = imread('~/code/visual-cortex/doc/markov-areas.png');
% white = sum(im,3) > 240*3;
% totalArea = sum(~white(:));
% 
%                                                                                                           15
% regions = {'V1', 'V2', 'V4', 'TEO', 'MT', 'TEpd', 'DP', 'STPc', '7a', 'STPr', 'STPi', 'PBr', '8l', '7m', '46d', '8m', '7B', '9/46d', '9/46v', '8B', '10', 'F7', 'F5', '5', 'F2', '24c', 'ProM', '2', 'F1'};
% if ~exist('areas', 'var')
%     areas = zeros(length(regions), 1);
% end
% for i = 1:length(regions)
%     if areas(i) == 0
%         mask = getRegionMask(im, regions{i});
%         areas(i) = sum(mask(:));
%         areas(i)
%     end
% end

% other variables: RF sizes (relative); RF extent (all vs. mostly foveal);
% list of feature maps; invariance (relative to RF size)

% fid = fopen('regions.csv','w');
% for i = 1:length(regions), fprintf(fid, '%s, %2.4f\n', regions{i}, areaFractions(i)); end
% fclose(fid)

im = imread('~/code/visual-cortex/doc/markov-conn.png');
n = 29;
square = round((size(im,1) + size(im,2)) / 2 / n);
centres = im((square/2)-1:square:end, square/2:square:end,:);
% imshow(centres)

im = imread('~/code/visual-cortex/doc/markov-levels.png');
n = 13;
square = round(size(im,1) / n);
levelColours = im(square/2:square:end,size(im,2)/2,:);
% levelColours = levelColours(4:2:end,1,:);
levels = [0 0 logspace(-6, 0, 25)];
levels = levels(2:2:end);
%imshow(levelColours)
levelColours = fliplr(squeeze(levelColours)');

centreLevels = zeros(size(centres,1), size(centres,2));
check = zeros(size(centres), 'uint8');
for i = 1:size(centres,1)
    for j = 1:size(centres,2)        
        if i == j
            ind = 0;
            check(i,j,2) = 255;
        else 
            centre = squeeze(centres(i,j,:));        
            distances = sum((repmat(double(centre), 1, size(levelColours,2)) - double(levelColours)).^2,1).^.5;
            ind = find(distances == min(distances), 1, 'first');
            centreLevels(i,j) = levels(ind);
            check(i,j,:) = levelColours(:,ind);
        end
    end
end

% figure
% subplot(1,2,1), imshow(centres)
% subplot(1,2,2), imshow(check)

% transpose connection matrix to more standard form (row per target)
centreLevels = centreLevels';

A = centreLevels > 0;
hierarchicalLevel = [1 2 3 4 3 5 6 6 6 6 6 0 7 6 7 7 6 0 8 7 0 0 8 6 0 0 0 0 0]; 
toKeep = find(hierarchicalLevel > 0);
hierarchicalLevel = hierarchicalLevel(toKeep);
A = A(toKeep, toKeep);
numConnections = sum(A(:))
maxTreeConnections = size(A,1)*(size(A,1)-1)/2

for i = 1:length(hierarchicalLevel)
    for j = 1:length(hierarchicalLevel)
        if hierarchicalLevel(i) < hierarchicalLevel(j) 
            A(i,j) = 0;
        end
    end
end
numConnections = sum(A(:))

% Here we manually delete from edges to remove cycles from the network ... 

% all:             'V1', 'V2', 'V4', 'TEO', 'MT', 'TEpd', 'DP', 'STPc', '7a', 'STPr', 'STPi', 'PBr', '8l', '7m', '46d', '8m', '7B', '9/46d', '9/46v', '8B', '10', 'F7', 'F5', '5', 'F2', '24c', 'ProM', '2', 'F1'};
% regions to keep: 'V1', 'V2', 'V4', 'TEO', 'MT', 'TEpd', 'DP', 'STPc', '7a', 'STPr', 'STPi', '8l' '7m' '46d' '8m' '7B' '9/46v' '8B' 'F5' '5'

name = {'V1', 'V2', 'V4', 'TEO', 'MT', 'TEpd', 'DP', 'STPc', '7a', 'STPr', 'STPi', '8l', '7m', '46d', '8m', '7B', '9/46v', '8B', 'F5', '5'};

A(8,7) = 0; %I think DP is higher-level (attention-related) that STP (polysensory)
A(9,7) = 0; %same for DP vs. 7a (egocentric object location)
A(10,7) = 0; %same for DP vs. STPr
A(11,7) = 0; %same for DP vs. STPi
A(:,7) = 0; %just made the same call for 8 (FEF), which is dubious -- let's treat DP as a dead end
A(9,8) = 0; %I think 7a is lower-level than STP (Felleman & Van Essen mum on this)
A(8,10) = 0; %I'll assume the ff path within the STPs is caudal to rostral
A(9,10) = 0; % as two up
A(11,10) = 0; % as two up
A(12,14) = 0; % think 46 (DLPFC) higher than 8 (which overlaps FEF but I don't know what the rest does)
A(12,15) = 0; A(15,12) = 0; %8l and 8m look like peers
A(15,14) = 0; % as two up
A(17,19) = 0; A(19,17) = 0; %I have these as output areas
A(18,14) = 0; % as two up
A(12,18) = 0; A(18,12) = 0; %look like peers
A(15,18) = 0; A(18,15) = 0; %look like peers
A(8,11) = 0; % assuming caudal to rostral in STPs
A(8,13) = 0; % tough call - I'll say 7m is higher (eye & hand pos)
A(8,9) = 0; % similar call as above
A(9,13) = 0; A(13,9) = 0; % seem like peers
A(13,16) = 0; A(16,13) = 0; % seem like peers
A(13,10) = 0; 
A(8,16) = 0; % tough call again
A(9,16) = 0; A(16,9) = 0; % seem like peers
A(20,16) = 0; %7 seems higher than 5
A(20,9) = 0; % as above
A(20,13) = 0; % as above
A(8,20) = 0; 
A(5,3) = 0; % based on latency (I think V4 is slower)

% channels, ypixels, xpixels, level (copied from spreadsheet)
data = [364	180	320	1;
    1414	72	128	2;
    16428	15	26.66666667	3;
    1045	36	64	4;
    5118	14.4	25.6	3;
    97424	3.6	6.4	5;
    26827	3.6	6.4	6;
    50742	2.88	5.12	6;
    76113	2.88	5.12	6;
    60670	2.88	5.12	6;
    76113	2.88	5.12	6;
    44	72	128	7;
    327	72	128	6;
    226	36	64	7;
    76	72	128	7;
    42	72	128	6;
    1	100	1	8;
    99	72	128	7;
    1	10	1	8;
    212	72	128	6];

chan = data(:,1);
[levelOffsets, centres] = NetworkPlot.plotLayers(data(:,3), data(:,2), chan, data(:,4), name);

%normalized to sum to 1 to account for rounding errors and eliminated
%connections ... 
centreLevels = centreLevels(toKeep, toKeep);
centreLevels(A == 0) = 0;
correctedLevels = centreLevels ./ repmat(sum(centreLevels,2), 1, size(centreLevels,2));
correctedLevels(1,:) = 0; %no ff inputs to V1
 
%mean # inputs from each area assuming 200 total (generous based on 10K;
%each unit approximates 50 spiking cells; no feedback or lateral
%connections)
meanInputs = 1000 * correctedLevels;

rfDia = [5 10 25 10 50 100 100 250 250 250 250 100 100 100 100 100 100 100 100 100];
convergenceRatio = zeros(length(rfDia));
for i = 1:length(rfDia)
    for j = 1:length(rfDia)
        toDia = rfDia(i);
        fromDia = rfDia(j);
        
        % this assumes half-RF stride, but connections are way too sparse
%         convergenceRatio(i,j) = (2*toDia/fromDia - 1)^2;
        
        % this assumes full-RF stride (also too sparse but can't do more)
        convergenceRatio(i,j) = (toDia/fromDia)^2;
    end
end

% note: interesting to plot mesh of convergence ratios -- huge where V1
% projects directly to higher-level areas

fullNum = convergenceRatio * (chan * chan');
sparseness = meanInputs ./ fullNum;

% mesh(sparseness.^(1/2))
nSourceChan = round(sparseness.^(1/4) .* repmat(chan, 1, size(sparseness,2)));
% mesh(nSourceChan)
% imagesc(nSourceChan>0)

rng(1)
for i = 1:size(nSourceChan,1)
    for j = 1:size(nSourceChan,2)
        if nSourceChan(i,j) > 1
            kw = toDia/fromDia;
            lkc = 10*log(nSourceChan(i,j));
            NetworkPlot.plotConnection(levelOffsets(j), centres(j), levelOffsets(i), centres(i), data(j,3), data(j,2), kw, lkc)
%             pause
        end
    end
end

for i = 1:size(nSourceChan,1)
    for j = [17 19]
        NetworkPlot.plotConnectionLine(levelOffsets(j), centres(j), levelOffsets(i), centres(i), data(j,3), data(j,2))
    end
end

set(gca, 'Visible', 'off')
set(gca, 'CameraPosition', [-2072.01 -3715.67 2383.07])
set(gcf, 'Position', [181 33 1097 773])
% toDia/fromDia

