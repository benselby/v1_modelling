%%Find fields of view and angle2pixel coefficients for color cameras
%%output:

%       FOV_color 1x2 vector [FOV_left, FOV_right]
%       angle2pix_color 1x2 vector [angle2pixel_left, angle2pixel_right]

function [FOV_color, angle2pix_color] = findFOV()
    endPixl = 1241;
    calibCamCam = loadCalibrationCamToCam('E:\Visual Cortex Model\full_model\Disparity\kitti\data\2011_09_26\calib_cam_to_cam.txt');
    [veloToCam, K] = loadCalibration('E:\Visual Cortex Model\full_model\Disparity\kitti\data\2011_09_26');

    X1_left = [-8.51025 0 10 1]'; 
    X2_left = [8.694 0 10 1]';
    %%
    % for left color camera (#02)
    R_rect_00 = calibCamCam.R_rect{1, 1};
    R_rect_00(4,4) = 1;
    P_rect_02 = calibCamCam.P_rect{1, 3};
    
    Y1_left = P_rect_02*X1_left;
    Y1_left = Y1_left /Y1_left(end);
    
    Y2_left = P_rect_02*X2_left;
    Y2_left = Y2_left /Y2_left(end);
    % RT_velo_to_cam = veloToCam{1,3};
    
    tan_angle_02 = abs((X1_left(1)- X2_left(1))./(2*X1_left(3)));
    angle_02 = 2*(180./pi) * atan(tan_angle_02);
    
    angle2pix_02 = endPixl/angle_02;
    
    %%
    X1_right = [-7.977515 0 10 1]'; 
    X2_right = [9.226 0 10 1]';
    %%
    % for left color camera (#02)
%     R_rect_00 = calibCamCam.R_rect{1, 1};
%     R_rect_00(4,4) = 1;
    P_rect_02 = calibCamCam.P_rect{1, 4};
    
    Y1_right = P_rect_02*X1_right;
    Y1_right = Y1_right /Y1_right(end);
    
    Y2_right = P_rect_02*X2_right;
    Y2_right = Y2_right /Y2_right(end);
    % RT_velo_to_cam = veloToCam{1,3};
    
    tan_angle_03 = abs((X1_right(1)- X2_right(1))./(2*X1_right(3)));
    angle_03 = 2*(180./pi) * atan(tan_angle_03);
    
    angle2pix_03 = endPixl/angle_03;
    
    FOV_color = [angle_02, angle_03];
    angle2pix_color = [angle2pix_02, angle2pix_03];
end