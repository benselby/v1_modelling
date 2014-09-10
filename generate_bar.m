function bar_img = generate_bar(theta, fsize)
    bar_img = zeros(fsize);
    bar_img(12:14, :) = 1;
    bar_img = imrotate(bar_img, theta, 'nearest', 'crop');
end