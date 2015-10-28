function gabor = gabor2(sig_x, sig_y, theta, phi, k, fsize)
    vals = linspace(-pi, pi, fsize);
    [xgrid, ygrid] = meshgrid(vals,vals);
    xy = xgrid*cos(theta) + ygrid*sin(theta);
    sine = sin(k*xy + phi);
    gaussian = exp(-(xgrid/(2*sig_x^2)).^2-(ygrid/(2*sig_y^2)).^2);
    
    gabor = gaussian*sine;
end