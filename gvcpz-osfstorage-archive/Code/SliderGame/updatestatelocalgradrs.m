function [posvec, dirvec, currtime, nexttime, nextind] = updatestatelocalgradrs(posvec, adjacency, dirvec, microstep, currtime, nexttime, switchtime, nextind, limvals)

% UPDATESTATELOCALGRADRS take small steps, repeating current direction if
% there is improvement, alter direction at larger random intervals

% First move to try to improve agreement
curragree = getagreelaopt(posvec, adjacency);
nextstate = crop(posvec + microstep*dirvec, limvals(1), limvals(2));
nextagree = getagreelaopt(nextstate, adjacency);
switchdir = nextagree < curragree;
dirvec(switchdir) = -dirvec(switchdir);
posvec = crop(posvec + microstep*dirvec, limvals(1), limvals(2));
currtime = currtime + microstep;

% Switch one player if the random time interval is up
if currtime > nexttime
    posvec(nextind) = -posvec(nextind);
    timecands = exprnd(switchtime, [1, length(posvec)]); % Sample exponential with mean switchtime
    nextlat = min(timecands);
    nextind = find(timecands == nextlat);
    nexttime = currtime + nextlat;
end





