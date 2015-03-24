%%Optical flow for left color camera
numFrames = 114;
im1 = imread('E:\Visual Cortex Model\full_model\Disparity\kitti\data\2011_09_26\2011_09_26_drive_0017_sync\image_02\data\0000000000.png');

uv = zeros(size(im1,1),size(im1,2),3,numFrames-1);

start = tic;
for i = 0:113
%     i
    if i < 10
        im1 = imread(['E:\Visual Cortex Model\full_model\Disparity\kitti\data\2011_09_26\2011_09_26_drive_0017_sync\image_02\data\000000000',num2str(i),'.png']);
        if i == 9
            im2 = imread(['E:\Visual Cortex Model\full_model\Disparity\kitti\data\2011_09_26\2011_09_26_drive_0017_sync\image_02\data\00000000',num2str(i+1),'.png']);
        else
            im2 = imread(['E:\Visual Cortex Model\full_model\Disparity\kitti\data\2011_09_26\2011_09_26_drive_0017_sync\image_02\data\000000000',num2str(i+1),'.png']);
        end
    elseif i < 100
            im1 = imread(['E:\Visual Cortex Model\full_model\Disparity\kitti\data\2011_09_26\2011_09_26_drive_0017_sync\image_02\data\00000000',num2str(i),'.png']);
        if i == 99
            im2 = imread(['E:\Visual Cortex Model\full_model\Disparity\kitti\data\2011_09_26\2011_09_26_drive_0017_sync\image_02\data\0000000',num2str(i+1),'.png']);
        else
            im2 = imread(['E:\Visual Cortex Model\full_model\Disparity\kitti\data\2011_09_26\2011_09_26_drive_0017_sync\image_02\data\00000000',num2str(i+1),'.png']);
        end
    else
            im1 = imread(['E:\Visual Cortex Model\full_model\Disparity\kitti\data\2011_09_26\2011_09_26_drive_0017_sync\image_02\data\0000000',num2str(i),'.png']);
%             im2 = imread(['E:\Visual Cortex Model\full_model\Disparity\kitti\data\2011_09_26\2011_09_26_drive_0017_sync\image_02\data\0000000',num2str(i+1),'.png']);
    end
     uv(:,:,:,i+1) = im1;
    figure(1), imshow(im1);
    set(gcf,'units','normalized','outerposition',[0 0 1 1])
    pause(.1)
end

movie(uv,1,10)
% exe_time = toc(start)
% % 
% % save('flows_0017.mat', 'uv', 'exe_time')
% 
% 
% %plot flow
% for i = 1:numFrames
%     i
%     figure(1), plotflow(uv(:,:,:,i));
%     pause(3);
% end