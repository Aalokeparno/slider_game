function tcount = report(tcount, base, lfactor)

% REPORT update on computation progress:  tcount = current counter value,
% base = how often to do display, lfactor = after how many displays to 
% make newline

% Eg report(dcount, 1, 10)

tcount = tcount + 1;
eltime = toc;
if nargin == 2
    if mod(tcount, base) == 0
        fprintf(' [%d:%.0f] ', tcount, eltime);
    end
    if (mod(tcount, 10*base) == 0)
        fprintf('\n');
    end
else
    if mod(tcount, base) == 0
        fprintf('[%d:%.0f]', tcount, eltime);
    end
    if mod(tcount, lfactor*base) == 0
        fprintf('\n');
    end
end