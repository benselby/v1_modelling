% Produces an MT neuron which responds to frames in KITTI
classdef mtNeuron < handle
    
    properties (GetAccess = public)
        recptiveFieldCentre = [];
        recptiveFieldSize = [];
        prefDirec = [];
        sigDirec = [];
        prefSpeed = [];
        sigSpeed = [];
        prefDisp = [];
        sigDisp = [];
        A = []; % A./(1+exp(-X))+ B;
        B = [];
    end
    
    properties (Access = private)        
%         mapDirecTuning = [];
%         mapSpeedTuning = [];
%         mapDispTuning = [];
    end
    
    methods (Access = public) 

        function mtNeuron = mtNeuron(recptiveFieldSize, recptiveFieldCentre, prefDirec, sigDirec, prefSpeed, sigSpeed, prefDisp, sigDisp) 
            mtNeuron.recptiveFieldCentre = recptiveFieldCentre;
            mtNeuron.recptiveFieldSize = recptiveFieldSize;
            
            mtNeuron.prefDirec = prefDirec;
            mtNeuron.sigDirec = sigDirec;
            
            mtNeuron.prefSpeed = prefSpeed;
            mtNeuron.sigSpeed = sigSpeed;
            
            mtNeuron.prefDisp = prefDisp;
            mtNeuron.sigDisp = sigDisp;
                                  
            mtNeuron.A = 1;
            mtNeuron.B = 0;
        end
       
        %%TO BE CHANGED
        function [mapframe, mapDirec, mapSpeed, mapDisp] = getRecepMaps(mtNeuron, frame, dirc, speed, disp)
            
            c = mtNeuron.recptiveFieldCentre;
            recepSize = mtNeuron.recptiveFieldSize;
            halfWidth = recepSize(1)/2-0.5;
            halfHeight = recepSize(2)/2-0.5;
                       

            mapframe = frame(c(1)-halfWidth:c(1)+halfWidth, c(2)-halfHeight:c(2)+halfHeight,:);
            mapDirec = dirc(c(1)-halfWidth:c(1)+halfWidth, c(2)-halfHeight:c(2)+halfHeight);
            mapSpeed = speed(c(1)-halfWidth:c(1)+halfWidth, c(2)-halfHeight:c(2)+halfHeight);
            mapDisp  = disp(c(1)-halfWidth:c(1)+halfWidth, c(2)-halfHeight:c(2)+halfHeight);
        end
        
        function [mapDirecTuning, mapSpeedTuning, mapDispTuning] = getMapsTuning(mtNeuron, mapframe, mapDirec, mapSpeed, mapDisp)
                        
            recepForm = size(mapframe);  %CAN BE A RECTANGLE TO MAKE LIFE EASY
                        
            cont = getContrast(mtNeuron, mapframe);
            
            mapDirecTuning = getDirecTuning(mtNeuron, recepForm, mapDirec(:));
            mapSpeedTuning = getSpeedTuning(mtNeuron, recepForm, cont, mapSpeed(:));
            mapDispTuning =  getDispTuning(mtNeuron, recepForm,  cont, mapDisp(:));
            
        end
        
        function rate = getRate(mtNeuron, mapDirecTuning, mapSpeedTuning, mapDispTuning, attentionGain)
                        
            recepKernel = getKernel(mtNeuron);                      
                       
            activation = attentionGain* recepKernel.* mapDirecTuning.* mapSpeedTuning.* mapDispTuning; 
            activation = sum(activation(:));
            
            rate = getNonlin(mtNeuron, activation);
        end
          
        %%Creating gabor functions based on "Spatial Freq selectivity of cells in Macaque visual cortex" by de Valois - 1981 
        %%and "Tutorial on Gabor Filters" by Javier R. Moveallen
        %%Contrast calculations based on "Contrast in complex images" by Eli Peli - 1990
        function contrast = getContrast(mtNeuron, mapframe)
           
           mapframe = double(rgb2gray(mapframe)); 
           u = 5; % 5 frequencies
           v = 4; % 4 orientations
           gaborArray = cell(u,v);
           
           m = mtNeuron.recptiveFieldSize(1);
           n = mtNeuron.recptiveFieldSize(2);
           
           l = zeros(m,n,v);
           c = zeros(m,n,v);
           
           halfWidth = n/2-0.5;
           halfHeight = m/2-0.5;
           x = repmat(-halfWidth:halfWidth, m, 1);
           y = repmat((-halfHeight:halfHeight)', 1, n);
           %F0 = 0.03589 /2;
           %R = 2.6390;
           fus = [1 1.4 2 2.8 5.6];
 
           for i = 1:u
%                fu = fmax/((sqrt(2))^(i-1));               
%                F = F0*R^(i-1); 
               F = fus(i) / 15.25; %cycle/degree to cycle/pixel multiplier comes from findFOV.m for left camera
               a = 0.9589 *F;
               b = 1.1866 *F;
    
%                a = 0.5589;
%                b = 0.69;
               for j = 1:v
                   tetav = ((j-1)/v)*pi;
%                    ((j-1)/v)*180;                  
                   xprime = x*cos(tetav)-y*sin(tetav);
                   yprime = x*sin(tetav)+y*cos(tetav);
                   gFilter  = ((a*b)/pi).*exp(-pi*(a^2.*(xprime.^2) + b^2.*(yprime.^2))).*exp(1i*2*pi*F*(x*cos(tetav)+y*sin(tetav)));
                   gaborArray{i,j} = gFilter;
        
               end
           end
           
           
           for j = 1:v
               for i = 1:u-1
                   l(:,:,j) = l(:,:,j) + abs(conv2(mapframe, gaborArray{i,j},'same'));
%                    figure(), imshow(abs(conv2(mapframe, gaborArray{i,j},'same')));
               end
               c(:,:,j) = (abs(conv2(mapframe, gaborArray{u,j},'same')))./l(:,:,j);
               figure(), imshow(c(:,:,j));
           end
           contrast = mean(c,3);
        end        
        
    end
    
    methods (Access = private)
        
        %%TO BE CHANGED
        
        function direcTuningOut = getDirecTuning(mtNeuron, recepForm, direcs)
            
%             sig = sigDirec;
            direcTuningOut = gaussmf(direcs,[mtNeuron.sigDirec mtNeuron.prefDirec]); 
            direcTuningOut = reshape(direcTuningOut, recepForm);
        end
        
        function speedTuningOut = getSpeedTuning(mtNeuron, recepForm, cont, speeds)
            
            c = mtNeuron.prefSpeed*cont; %%To be determined             
            sig = mtNeuron.sigSpeed;
            speedTuningOut = gaussmf(speeds,[sig c]); 
            speedTuningOut = reshape(speedTuningOut, recepForm);
        end
        
        function dispTuningOut = getDispTuning(mtNeuron, recepForm, cont, disps)
            
            c = mtNeuron.prefDisp*cont; %%To be determined
            sig = mtNeuron.sigDisp;
            dispTuningOut = gaussmf(disps,[sig c]); 
            dispTuningOut = reshape(dispTuningOut,recepForm);
        end
            
         %CAN BE A STATIC METHOD LATER ON?!   
         function  rate = getNonlin(mtNeuron, activation)
                
             rate = (mtNeuron.A)./(1+exp(-activation))+ mtNeuron.B;
         end
    
         %AFTER OPTIMIZATION CAN BE A STATIC METHOD
         function  recepKernel = getKernel(mtNeuron)
            
             recepKernel = .001 * rand(mtNeuron.recptiveFieldSize(1),mtNeuron.recptiveFieldSize(2));
             
         end
    end
    
    methods (Static)
     
       
        
    end
end
