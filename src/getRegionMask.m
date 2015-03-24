function mask = getRegionMask(im, area)
    % Segments a colour region around a given point. 
    % 
    % im: a cortical flatmap image with different regions in different
    %   colours
    % area: name of an area to find a mask for 
    % 
    % mask: image-sized matrix with ones over area and zeros elsewhere

    threshold = 25;

    figure(1), set(gcf, 'Position', [41         265        1294         541])
    subplot(1,2,1), imshow(im), title(area)
    [X,Y] = ginput; 

    X = round(X);
    Y = round(Y);

    h = size(im,1);
    w = size(im,2);

    mask = zeros(size(im,1), size(im,2));
    mask(Y, X) = 1;

    ref = double(repmat(im(Y, X, :), h, w, 1));
    diff = double(im) - ref;
    close = sum(diff.^2,3).^.5 < threshold;

    goodCandidates = getCandidates(mask, h, w) & close;
    while max(goodCandidates(:)) 
        mask = max(mask, goodCandidates);
        goodCandidates = getCandidates(mask, h, w) & close;
    end

    subplot(1,2,2), imshow(mask)
end

function candidates = getCandidates(mask, h, w)
    % shift mask left, right, up, down to get neighbours
    candidates = [mask(:,2:end) zeros(h,1)];
    candidates = max(candidates, [zeros(h,1) mask(:,1:end-1)]);
    candidates = max(candidates, [mask(2:end,:); zeros(1,w)]);
    candidates = max(candidates, [zeros(1,w); mask(1:end-1,:)]);
    candidates = candidates & ~mask;
end
