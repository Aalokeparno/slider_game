function [curragree, totalagree] = getagreelaopt(currstate, adjacency)

% GETAGREE

curragree = abs(adjacency * currstate) ./ sum(adjacency, 2);

totalagree = min(curragree);
%totalagree = abs(sum(currstate))/length(currstate);  % = curragree with full connectivity (same val for all players)