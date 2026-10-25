function val = rpick(nn, dsize)

% Returns an object of form dsize containing random integers between 1 and nn, uniform distribution

if nargin < 2
    dsize = 1;
end

val = floor(nn*rand(dsize)) + 1;
