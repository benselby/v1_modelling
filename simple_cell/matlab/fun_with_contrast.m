clear all
close all

rate     = [5    10   22   30  40  80  90  100 ];
contrast = [0.03 0.06 0.15 0.2 0.3 0.5 0.7 1];

theta = pi/4;
fsize = 25;
k = 4;
contrast_bank = zeros(fsize, fsize, numel(contrast));
grating = generate_grating(k, theta, fsize);
for i=1:numel(contrast)
    contrast_bank(:,:,i) = contrast(i)*(grating-0.5) + 0.5;
    figure()
    imshow(contrast_bank(:,:,i))
    tit=sprintf('Contrast = %.2f', contrast(i));
    title(tit)
end

c = zeros(numel(contrast),1);
for f=1:numel(contrast)
    c(f) = ( max(contrast_bank(:,:,f) ) - min(contrast_bank(:,:,f) ) ) / (max(contrast_bank(:,:,f)) + min(contrast_bank(:,:,f)));
end

figure()
plot(contrast, c)
