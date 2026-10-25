function wstatus = winningorthant(currstate)

% WINNING determine if currstate is a winning state

if numel(currstate) == abs(sum(sign(currstate)))
    wstatus = 1;
else
    wstatus = 0;
end