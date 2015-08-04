function bar_img = generate_bar(theta, fsize)
    bar_img = zeros(fsize);
    bar_img(floor(fsize/2-1):ceil(fsize/2+1), :) = 1;
    bar_img = imrotate(bar_img, theta, 'nearest', 'crop');
end