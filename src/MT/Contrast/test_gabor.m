
u = 5;
v = 4;

m = 128;
n = 128;

img = imread('E:\Visual Cortex Model\full_model\Disparity\kitti\data\2011_09_26\2011_09_26_drive_0017_sync\image_02\data\0000000000.png');

gaborArray = gaborFilterBank2(u,v,m,n);

featureVector = gaborFeatures(img,gaborArray,1,1);

for j = 1:v
    for i = 1:u
        figure('NumberTitle','Off','Name',['Freq = ',num2str(i), ' Orientation = ', num2str((j-1)*45)]), imshow(featureVector(:,:,i,j))
        pause(2);
    end
end