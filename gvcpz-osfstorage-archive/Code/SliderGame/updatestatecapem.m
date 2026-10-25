function [posvec, dirvec, nextind, nextlat] = updatestatecapem(posvec, dirvec, adjacency, meanstep, agreethresh, posthresh, limvals)

% UPDATESTATECAPEM update state under continuous APEM

% Get agreement
curragree = getagreelaopt(posvec, adjacency);

% Choose time interval and candidate for directional switch
timecands = exprnd(meanstep, [1, length(posvec)]); % Sample exponential with mean meanstep
nextlat = min(timecands);
nextind = find(timecands == nextlat);
if length(nextind) > 1
    fprintf('\nWarning:  found multiple minimum latencies. Choosing one randomly.');
    nextind = nextind(randi(length(nextind)));
end
if (curragree(nextind) <= agreethresh) && (abs(posvec(nextind)) >= posthresh)
    dirvec(nextind) = -dirvec(nextind);
end

% Follow current direction, updating time history
posvec = crop(posvec + nextlat*dirvec, limvals(1), limvals(2));

% nextind
% posvec'
% dirvec'
% keyboard;


