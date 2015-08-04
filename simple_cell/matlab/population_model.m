%
% Population Model
% Generates a layer of V1 simple cells with lateral connections

clear all
close all

plotting = true;

% size of the layer in number of neurons 
row = 10;
col = 10;
depth = 1;

% Alternatively, we could specify the size of the layer by mm:
density = 10; % # of neurons/mm
x_len = 2.5; % mm
y_len = 2.5;
depth = 1;

X = 0:1/(density):x_len;
Y = 0:1/(density):y_len;

% Develop eccentricity and azimuth of the visual field based on cortical
% surface area
lambda = 12; % mm, from Dayan and Abbott pg 57
ecc0 = 1; % deg, from Dayan and Abbott pg 57
ecc = (exp(X./lambda) - 1)/ecc0;
a = -180*(ecc0+ecc).*Y ./ (lambda*ecc*pi);

[egrid, agrid] = meshgrid(ecc, a); 

% Pre-allocate layer with v1 neurons:
v1_layer(numel(X), numel(Y), depth) = v1_neuron();

N = numel(v1_layer); % number of neurons in layer

% Get property distributions from JSON specification
max_fr = 100;
min_fr = 10;
max_rate = min_fr + (max_fr-min_fr)*rand(N);

% Uniform orientation preference
orient_pref = 360*rand(N,1);

% (very) approximate orientation bandwidth from Devalois et al, 1982
u_band = 40;
sig_band = 10;
bandwidth = u_band + sig_band*randn(N,1);

% RF size should scale with eccentricity - some fxn of neuron index
pix_deg = 25; % pixels per degree (horizontal), based on camera configuration/target dataset

% From Hubel and Wiesel, 1974, fig 6A:
data = [0.96415, 0.31680;
3.97960, 0.41742;
6.96179, 0.62719;
8.55094, 0.58242;
10.04219, 0.94376;
18.11431, 1.00511;
22.19939, 1.33566];

p = polyfit(data(:,1), data(:,2), 1);

rf_size_map = p(1)*egrid + p(2); % RF size in degrees

rf_pix_map = rf_size_map*pix_deg; % convert RF sizes to pixels

% Add random variation to RF sizes:


%% Create orientation preference map:
img_str = '../figures/v1-topology-blasdel-figure6.png';
op_map = zeros(size(v1_layer));
op_scale = 108; % pixels per mm, specified by the figure
pix_n = round(op_scale/density); % pixels/neuron, for traversing labelled map
n_colours = 6; % number of colours in the image
op_range = 0:180/n_colours:180-180/n_colours; % range of orientation preferences (degrees)

% Run k-means on the image
pixel_labels = segment_kmeans(img_str, n_colours);

for i=1:size(v1_layer,1)
   for j=1:size(v1_layer,2)
       op_map(i,j) = op_range( pixel_labels(i*pix_n, j*pix_n) );
   end
end

% Add random variation to orientation preferences

% Segment a topological map image from Blasdel, 1992 using K means colour
% segmentation:


%% Create ocular dominance map:
od_map = zeros(size(v1_layer));


%% Use the specified maps to populate the layer:
% tic
% for i=1:N 
%     v1_layer(i) = v1_neuron(max_rate(i), bandwidth(i), op_map(i), rf_map(i));
% %     v1_layer(i) = v1_neuron(max_rate(i), bandwidth(i), orient_pref(i), fsize(i));
% end
% toc

% Now play some frames from the KITTI video and record the firing rates of
% each neuron

%% Plotting
if plotting==true
   
    figure()
    hold on
    scatter(data(:,1), data(:,2))
    plot(ecc, rf_size_map(1,:), 'r')
    title('Receptive Field Size vs Eccentricity')
    xlabel('Eccentricity (deg)')
    ylabel('RF Size (deg)')
    legend('Hubel and Wiesel Data, 1974', 'Model')
    
    figure()
    imshow(pixel_labels,[])
    title('Segmented Orientation Preference Topology')
end