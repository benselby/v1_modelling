% A test script for colour-segmentation of V1 topological images
% Focusing on Blasdel's differential imaging of orientation columns (1992)
%
% Source:
% http://www.mathworks.com/help/images/examples/color-based-segmentation-us
% ing-k-means-clustering.html

clear all
close all

he = imread('../figures/v1-topology-blasdel-figure6.png');
figure()
imshow(he), title('Original image from Blasdel, 1992');

%% Convert image from RGB to L*a*b* colour space
cform = makecform('srgb2lab');
lab_he = applycform(he,cform);
ab = double(lab_he(:,:,2:3));
nrows = size(ab,1);
ncols = size(ab,2);
ab = reshape(ab,nrows*ncols,2);

%% Run K means clustering to identify the 6 different colours 
nColors = 6;
% repeat the clustering 3 times to avoid local minima
[cluster_idx, cluster_center] = kmeans(ab,nColors,'distance','sqEuclidean', ...
                                      'Replicates',3);
pixel_labels = reshape(cluster_idx,nrows,ncols);
figure()
imshow(pixel_labels,[]), title('image labeled by cluster index');
segmented_images = cell(1,3);
rgb_label = repmat(pixel_labels,[1 1 3]);

%% Plot individual segments of original image
% for k = 1:nColors
%     color = he;
%     color(rgb_label ~= k) = 0;
%     segmented_images{k} = color;
%     
%     figure()
%     title_str = sprintf('objects in cluster %d', k);
%     imshow(segmented_images{k}), title(title_str);
% end



