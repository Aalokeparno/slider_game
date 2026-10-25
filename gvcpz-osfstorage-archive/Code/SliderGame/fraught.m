function [fstatus, impinds] = fraught(currpos, adjacency)

% FSTATUS determine if currpos is a fraught state of the network
% specified by adjacency

nplayers = length(currpos);
impinds = []; % To be:  indices of players who can improve by switching

if winningorthant(currpos)
    fstatus = 0;
elseif prod(currpos) == 0
    fstatus = 0;
else 
    fstatus = 1;
    for pcount = 1:nplayers
        currsign = sign(currpos(pcount));
        gfinds = sign(currpos) == currsign; % Identify players that agree in sign
        geinds = 1 - gfinds;
        nbrs = adjacency(pcount, :);
        nbrs(pcount) = 0; % Don't count self as a neighbor for this computation
        fsum = abs(nbrs * (currpos .* gfinds));
        esum = abs(nbrs * (currpos .* geinds));
        if esum > fsum
            fstatus = 0;
            impinds = [impinds, pcount];
        end
    end
end
