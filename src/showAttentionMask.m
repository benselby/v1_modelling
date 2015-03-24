% load('attentionMask_0_9.mat')
% load('attentionMask_10_19.mat')
% load('attentionMask_20_29.mat')
% load('attentionMask_30_39.mat')
% load('attentionMask_40_49.mat')
% load('attentionMask_50_59.mat')
% load('attentionMask_60_99.mat')
% load('attentionMask_100_113.mat')

load('attentionMask_0_113.mat')


for i = 1 : size(attentionMask,3)
    imshow(attentionMask(:,:,i));
    i
    pause(1);

end
