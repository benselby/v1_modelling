%%Find fields of view and angle2pixel coefficients for color cameras
%%output:

%       FOV_color 1x2 vector [FOV_left, FOV_right]
%       angle2pix_color 1x2 vector [angle2pixel_left, angle2pixel_right]

function [FOV_color, angle2pix_color] = findFOV2()

    calibCamCam = loadCalibrationCamToCam('E:\Visual Cortex Model\full_model\Disparity\kitti\data\2011_09_26\calib_cam_to_cam.txt');
    [veloToCam, K] = loadCalibration('E:\Visual Cortex Model\full_model\Disparity\kitti\data\2011_09_26');

    %Top left corner pixel 
    endPixl = 1241;
    Y1 = [1 0 1]'; 
    Y2 = [endPixl 0  1]';
    %%
    % for left color camera (#02)
    R_rect_00 = calibCamCam.R_rect{1, 1};
    R_rect_00(4,4) = 1;
    P_rect_02 = calibCamCam.P_rect{1, 3};

    % RT_velo_to_cam = veloToCam{1,3};

    % total = P_rect_02 * R_rect_00 * RT_velo_to_cam;
    total = P_rect_02 * 1;

    invTotal = pinv(total);

    X1 = invTotal * Y1;
    X1_left = X1/X1(end); 

    X2 = invTotal * Y2;
    X2_left = X2/X2(end);

    tan_angle_02 = abs((X1_left(1)- X2_left(1))./(2*X1_left(3)));
    angle_02 = 2*(180./pi) * atan(tan_angle_02);

    angle2pix_02 = endPixl/angle_02;
    %%
    % for right color camera (#03)
    P_rect_03 = calibCamCam.P_rect{1, 4};

    % RT_velo_to_cam = veloToCam{1,4};

    % total = P_rect_02 * R_rect_02 * RT_velo_to_cam;
    total = P_rect_03 * R_rect_03;

    invTotal = pinv(total);

    X1 = invTotal * Y1;
    X1_right = X1/X1(end); 

    X2 = invTotal * Y2;
    X2_right = X2/X2(end);

    tan_angle_03 = abs((X2_right (2)- X1_right (2))./(2*X1_right(3)));
    angle_03 = 2*(180./pi) * atan(tan_angle_03);

    angle2pix_03 = endPixl/angle_03;
    
    FOV_color = [angle_02, angle_03];
    angle2pix_color = [angle2pix_02, angle2pix_03];
end