function out = gaussian_ori(x,y,sigma)
%     return np.abs( np.mod( x - y + 90, 180) - 90 )
% 
% def G(x,y,sigma):
%     return np.exp(-1*diff(x,y)**2/(2*sigma**2))
    diff = abs( mod( x - y + 90, 180) - 90 );
    out = exp(-1 * diff.^2/(2*sigma^2) );
end