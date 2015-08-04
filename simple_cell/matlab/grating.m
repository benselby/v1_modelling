% Generate a sinusoidal grating
function gr=generate_grating(freq, theta, fsize)
    gr = zeros(fsize);
    for i=1:fsize
       for j=1:fsize
           x = i - floor(fsize/2) - 1;
           y = j - floor(fsize/2) - 1;
           Xj = x*cos(theta) - y*sin(theta);
           gr(i,j) = mat2gray(cos(freq*Xj));  
       end
    end 
end