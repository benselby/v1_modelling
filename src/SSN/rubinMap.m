% Produces an estimate of orientations from Rubin et al. 2015 figure 6. 

try 
    key = imread('rubin-orientation-key.png');
catch E
    warning('You have to run this code from the directory that has rubin-orientation-key.png')
    rethrow(E)
end

key = uint8(mean(key, 1));
key = squeeze(key);
key = uint8(conv2(ones(4,1)/4, 1, key, 'same'));
orientations = 0:180/(size(key,1)-1):180;

op = imread('rubin-orientation.png');

% note there are 75x75 squares but their dimensions aren't integers 
s = [size(op,1)/75 size(op,2)/75];

vcentres = round(s(1)/2:s(1):size(op,1)-s(1)/2);
hcentres = round(s(2)/2:s(2):size(op,2)-s(2)/2);

clean = zeros(75,75,3,'uint8');
for i = 1:length(vcentres)
    for j = 1:length(hcentres)
        region = op(vcentres(i)+(-2:2), hcentres(j)+(-2:2), :);
        clean(i,j,1) = mean(mean(region(:,:,1)));
        clean(i,j,2) = mean(mean(region(:,:,2)));
        clean(i,j,3) = mean(mean(region(:,:,3)));
    end
end

map = zeros(75,75);
for i = 1:length(vcentres)
    for j = 1:length(hcentres)
        differences = repmat(squeeze(double(clean(i,j,:)))', size(key,1), 1) - double(key);
        distances = sum(differences.^2, 2);
        [mindist,ind] = min(distances);
        map(i,j) = orientations(ind);
    end
end

figure, set(gcf, 'Position', [400 400 800 600])

subplot(2,2,1), imshow(clean), title('cleaned figure', 'FontSize', 18)

subplot(2,2,2), hold on
scatter3(key(:,1), key(:,2), key(:,3))
for i = 1:10:size(key,1)
    text(double(key(i,1)), double(key(i,2)), double(key(i,3)), sprintf('%i', uint8(orientations(i))))
end
clean1 = clean(:,:,1); clean2 = clean(:,:,2); clean3 = clean(:,:,3);
scatter3(clean1(:), clean2(:), clean3(:), 'k.')
set(gca, 'FontSize', 18)
xlabel('red', 'FontSize', 18)
ylabel('green', 'FontSize', 18)
zlabel('blue', 'FontSize', 18)

title('scatter of key and map values', 'FontSize', 18)

subplot(2,2,3), mesh(map), title('estimated orientations', 'FontSize', 18)
set(gca, 'FontSize', 18)
xlabel('x', 'FontSize', 18)
ylabel('y', 'FontSize', 18)
zlabel('orientation', 'FontSize', 18)

subplot(2,2,4), hist(map(:), 100), title('histogram of estimated orientations', 'FontSize', 18)
set(gca, 'FontSize', 18)
xlabel('orientations', 'FontSize', 18)
ylabel('counts', 'FontSize', 18)

save('orientation-map.mat', 'map')

