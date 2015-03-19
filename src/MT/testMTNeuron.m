%%Load disparity 
load('E:\Visual Cortex Model\full_model\Disparity\\disp_0017.mat')
%%Load flow 
load('E:\Visual Cortex Model\full_model\Optical flow\flows_0017.mat')

%%FROM JSON
recptiveFieldSize = [49, 59];
recptiveFieldCentre = [200,600];
prefDirec = 0;
sigDirec = 20;
prefSpeed = 32;
sigSpeed = 20;
prefDisp = 1;
sigDisp = .5;

attentionGain = 1;

for numFrame = 1:112
    
    frame = imread('E:\Visual Cortex Model\full_model\Disparity\kitti\data\2011_09_26\2011_09_26_drive_0017_sync\image_02\data\0000000000.png');
    figure(),imshow(frame);
    hold on, rectangle('Position',[recptiveFieldCentre(2)-(recptiveFieldSize(2)-1)/2 recptiveFieldCentre(1)-(recptiveFieldSize(1)-1)/2 recptiveFieldSize(2) recptiveFieldSize(1)]);
    %FROM FLOW FIELD
    flow = uv(:,:,:,numFrame);
    
    [dirc, speed] = cart2pol(flow(:,:,1),flow(:,:,2));
    dirc = 180./pi.*dirc;
    

    MT1 = mtNeuron(recptiveFieldSize, recptiveFieldCentre, prefDirec, sigDirec, prefSpeed, sigSpeed, prefDisp, sigDisp); 

    [mapframe, mapDirec, mapSpeed, mapDisp] = getRecepMaps(MT1, frame, dirc, speed, disp(:,:,numFrame));
    figure(),imshow(mapframe)
    figure(),imshow(mapDisp)
    
    
    cont = getContrast(MT1, mapframe);
    figure(), imshow(cont);
    
    [mapDirecTuning, mapSpeedTuning, mapDispTuning] = getMapsTuning(MT1, mapframe, mapDirec, mapSpeed, mapDisp);

    rate = getRate(MT1, mapDirecTuning, mapSpeedTuning, mapDispTuning, attentionGain)

end