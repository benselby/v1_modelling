% Attention polygon

classdef AttentionPolygon < handle
    
    properties (Access = public)
        img = [];
        snapTolerance = [];
        handle = [];
        polygons = cell(1,0);
    end
    
    methods (Access = public)
        function ap = AttentionPolygon(img, snapTolerance)
            ap.img = img;
            ap.snapTolerance = snapTolerance;
        end
        
        function addPolygon(ap)
            show(ap);
            
            [X,Y] = ginput; 
            
            if ~isempty(ap.polygons)
                firstPolygon = ap.polygons{1};
                for i = 1:length(X)
                    p = [X(i); Y(i)];
                    snapped = snap(firstPolygon.X, firstPolygon.Y, p, ap.snapTolerance);
                    X(i) = snapped(1); Y(i) = snapped(2);
                end
            end
            
            angles = zeros(size(X,1)-1,1);
            for i = 1:length(angles)
                difference = [X(i+1); Y(i+1)] - [X(1); Y(1)];
                angles(i) = atan(difference(2)/difference(1));
            end
            angles(angles < 0) = angles(angles < 0) + pi;
            [~,I] = sort(angles);
            ind = [1; 1+I];
            X = X(ind); Y = Y(ind);
            p = struct('X', X, 'Y', Y);
            ap.polygons{length(ap.polygons)+1} = p;
            
            drawPolygon(X,Y);
        end
        
        function show(ap)
            if isempty(ap.handle) || ~ishandle(ap.handle)
                ap.handle = imshow(ap.img);
                hold on
                for i = 1:length(ap.polygons)
                    X = ap.polygons{i}.X;
                    Y = ap.polygons{i}.Y;
                    drawPolygon(X,Y);
                end
            end            
        end
    end
    
end

function drawPolygon(X,Y)
    colours = {'r', 'g', 'b', 'c', 'm', 'y'};
    for i = 3:length(X)
        Xi = [X(1); X(i-1); X(i)];
        Yi = [Y(1); Y(i-1); Y(i)];
        colour = colours{1+mod(i, length(colours))};
        fill(Xi,Yi,colour,'FaceAlpha',.5)
    end
end

function snapped = snap(X, Y, p, tolerance)
    % X: list of horizontal coords in polygon
    % Y: list of vertical coords in polygon
    % p: point to snap
    % tolerance: max distance to move point
    % 
    % snapped: p snapped to closest edge if distance within tolerance
    
    snapped = p;
    
    nedges = length(X); %outside edges only
    snapDistances = zeros(nedges,1);
    snapPoints = zeros(2,nedges);
    for i = 1:nedges
        ind = [i,1+mod(i,nedges)];
        p1 = [X(ind(1)); Y(ind(1))]; %points that make up edge
        p2 = [X(ind(2)); Y(ind(2))];
        distanceAlongLine = (p-p1)'*(p2-p1)/norm(p2-p1);
        if distanceAlongLine < 0
            distance = norm(p-p1);
            point = p1;
        elseif distanceAlongLine > norm(p2-p1)
            distance = norm(p-p2);
            point = p2;
        else
            point = p1 + distanceAlongLine*(p2-p1)/norm(p2-p1);
            distance = norm(p-point);
        end
        snapDistances(i) = distance;
        snapPoints(:,i) = point;
    end
    
    if max(snapDistances <= tolerance)
        ind = find(snapDistances == min(snapDistances), 1, 'first');
        snapped = snapPoints(:,ind);
    end
end
