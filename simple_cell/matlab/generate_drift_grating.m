% A function for generating drifting gratings for temporal frequency models
% Generates a three-dimensional matrix of a oriented grating drifting in a
% perpendicular direction for T seconds
function gr = generate_drift_grating(T, dt, sfreq, tfreq, theta, fsize)

gr = zeros(fsize, fsize, round(T/dt));
phase = 0;

for i=1:round(T/dt)
    phase = phase + 2*pi*dt*tfreq;
    gr(:,:,i) = generate_grating(sfreq, theta, fsize, phase);
end

end